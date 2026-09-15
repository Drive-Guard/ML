"""
Cliente de ingestão do DriveGuard — ponte entre o dispositivo embarcado e a AWS.

Envia as leituras de fadiga para o endpoint `POST /v1/eventos` do API Gateway
provisionado em https://github.com/Drive-Guard/Infra.

Três decisões que definem o desenho desta classe:

1. **O alerta nunca depende da rede.** O classificador roda no veículo e o
   alerta é local. Este cliente só cuida do que vem depois: mandar a telemetria
   para a nuvem. Se a rede cair, o motorista continua protegido.

2. **Envio em lote, não por leitura.** Um caminhão em estrada passa horas sem
   sinal. As leituras se acumulam num buffer e sobem juntas quando há rede,
   o que também reduz o custo de requisição no API Gateway.

3. **Nenhum dado pessoal sai do veículo.** `motorista_id` e `placa` são
   convertidos em SHA-256 antes de qualquer envio, e nenhuma imagem facial é
   transmitida — só as métricas numéricas (LGPD / privacy by design).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import requests

logger = logging.getLogger(__name__)


def hash_identificador(valor: str) -> str:
    """
    Converte um identificador pessoal (CPF, matrícula, placa) em SHA-256.

    É o mesmo formato que o schema `silver` espera em `motorista_hash`,
    `placa_hash` e `cnpj_hash`. Irreversível: a nuvem nunca recebe o valor
    original.
    """
    return hashlib.sha256(valor.strip().encode("utf-8")).hexdigest()


class DriveGuardClient:
    """Acumula leituras e as envia em lote para a API de ingestão."""

    def __init__(
        self,
        api_url: str | None = None,
        api_key: str | None = None,
        device_id: str | None = None,
        motorista_id: str | None = None,
        veiculo_placa: str | None = None,
        tipo_veiculo: str = "caminhao",
        tamanho_lote: int = 60,
        timeout: int = 10,
        max_tentativas: int = 3,
        buffer_offline: str | Path = "data/buffer_offline.jsonl",
    ) -> None:
        self.api_url = (api_url or os.environ.get("DRIVEGUARD_API_URL", "")).rstrip("/")
        self.api_key = api_key or os.environ.get("DRIVEGUARD_API_KEY", "")
        self.device_id = device_id or os.environ.get("DRIVEGUARD_DEVICE_ID", "dg-edge-0001")

        motorista = motorista_id or os.environ.get("DRIVEGUARD_MOTORISTA_ID", "")
        placa = veiculo_placa or os.environ.get("DRIVEGUARD_PLACA", "")

        self.motorista_hash = hash_identificador(motorista) if motorista else None
        self.veiculo_hash = hash_identificador(placa) if placa else None
        self.tipo_veiculo = tipo_veiculo

        self.tamanho_lote = tamanho_lote
        self.timeout = timeout
        self.max_tentativas = max_tentativas

        self.buffer_offline = Path(buffer_offline)
        self.buffer_offline.parent.mkdir(parents=True, exist_ok=True)

        self._leituras: list[dict[str, Any]] = []
        self._alertas: list[dict[str, Any]] = []
        self._lock = threading.Lock()

        self.enviados = 0
        self.falhas = 0

        if not self.api_url:
            logger.warning(
                "DRIVEGUARD_API_URL nao configurada: o cliente roda em modo offline "
                "e so grava no buffer local."
            )

    # ------------------------------------------------------------------ estado

    @property
    def configurado(self) -> bool:
        return bool(self.api_url and self.api_key and self.motorista_hash)

    def health(self) -> bool:
        """Checa o endpoint /health (não exige API Key) antes de montar um lote."""
        if not self.api_url:
            return False
        try:
            r = requests.get(f"{self.api_url}/health", timeout=self.timeout)
            return r.status_code == 200
        except requests.RequestException:
            return False

    # --------------------------------------------------------------- acumulacao

    def registrar_leitura(
        self,
        ear: float,
        mar: float,
        score_fadiga: float,
        estado: str,
        perclos: float | None = None,
        blink_rate: float | None = None,
        duracao_olhos_fechados_ms: int | None = None,
        head_pitch: float | None = None,
        head_yaw: float | None = None,
        head_roll: float | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
        velocidade_kmh: float | None = None,
        registrado_em: datetime | None = None,
    ) -> None:
        """Adiciona uma leitura ao buffer. Envia sozinho ao atingir o lote."""
        ts = registrado_em or datetime.now(timezone.utc)

        leitura = {
            "registrado_em": ts.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            "ear": _arredondar(ear, 4),
            "mar": _arredondar(mar, 4),
            "perclos": _arredondar(perclos, 4),
            "blink_rate": _arredondar(blink_rate, 2),
            "duracao_olhos_fechados_ms": duracao_olhos_fechados_ms,
            "head_pitch": _arredondar(head_pitch, 2),
            "head_yaw": _arredondar(head_yaw, 2),
            "head_roll": _arredondar(head_roll, 2),
            "score_fadiga": _arredondar(score_fadiga, 2),
            "estado": estado,
            "latitude": latitude,
            "longitude": longitude,
            "velocidade_kmh": _arredondar(velocidade_kmh, 1),
        }
        leitura = {k: v for k, v in leitura.items() if v is not None}

        with self._lock:
            self._leituras.append(leitura)
            precisa_enviar = len(self._leituras) >= self.tamanho_lote

        if precisa_enviar:
            self.enviar()

    def registrar_alerta(
        self,
        tipo: str,
        gravidade: str,
        score_fadiga: float,
        mensagem: str | None = None,
        disparado_em: datetime | None = None,
    ) -> None:
        """
        Registra um alerta já emitido no veículo.

        O alerta sonoro/visual acontece antes desta chamada — aqui ele só vira
        telemetria para o dashboard do gestor de frota.
        """
        ts = disparado_em or datetime.now(timezone.utc)
        with self._lock:
            self._alertas.append(
                {
                    "disparado_em": ts.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
                    "tipo": tipo,
                    "gravidade": gravidade,
                    "score_fadiga": _arredondar(score_fadiga, 2),
                    "mensagem": mensagem,
                }
            )

    # ------------------------------------------------------------------- envio

    def enviar(self, forcar: bool = False) -> bool:
        """
        Envia o buffer atual. Devolve True se a API aceitou o lote.

        Em caso de falha, o lote vai para o buffer offline em disco em vez de
        ser perdido, e sobe na próxima chamada de `drenar_buffer_offline`.
        """
        with self._lock:
            if not self._leituras and not forcar:
                return True
            leituras, self._leituras = self._leituras, []
            alertas, self._alertas = self._alertas, []

        if not leituras:
            return True

        payload = self._montar_payload(leituras, alertas)

        if not self.configurado:
            self._gravar_offline(payload)
            return False

        if self._post(payload):
            self.enviados += len(leituras)
            return True

        self.falhas += len(leituras)
        self._gravar_offline(payload)
        return False

    def _montar_payload(self, leituras: list, alertas: list) -> dict[str, Any]:
        return {
            "device_id": self.device_id,
            "schema_version": "1.0",
            "motorista_hash": self.motorista_hash,
            "veiculo_hash": self.veiculo_hash,
            "tipo_veiculo": self.tipo_veiculo,
            "enviado_em": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "leituras": leituras,
            "alertas": alertas,
        }

    def _post(self, payload: dict[str, Any]) -> bool:
        headers = {"x-api-key": self.api_key, "Content-Type": "application/json"}
        espera = 1.0

        for tentativa in range(1, self.max_tentativas + 1):
            try:
                r = requests.post(
                    f"{self.api_url}/eventos",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )

                if r.status_code == 202:
                    corpo = r.json()
                    logger.info(
                        "Lote aceito: %s leituras, ingest_id=%s",
                        corpo.get("leituras_recebidas"),
                        corpo.get("ingest_id"),
                    )
                    return True

                # 4xx é payload malformado: repetir não resolve.
                if 400 <= r.status_code < 500 and r.status_code != 429:
                    logger.error(
                        "Lote rejeitado (HTTP %s): %s", r.status_code, r.text[:500]
                    )
                    return False

                logger.warning(
                    "API respondeu HTTP %s (tentativa %d/%d)",
                    r.status_code, tentativa, self.max_tentativas,
                )
            except requests.RequestException as exc:
                logger.warning(
                    "Falha de rede (tentativa %d/%d): %s",
                    tentativa, self.max_tentativas, exc,
                )

            if tentativa < self.max_tentativas:
                time.sleep(espera)
                espera *= 2

        return False

    # ---------------------------------------------------------- buffer offline

    def _gravar_offline(self, payload: dict[str, Any]) -> None:
        try:
            with self.buffer_offline.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
            logger.info(
                "Lote guardado no buffer offline (%d leituras).", len(payload["leituras"])
            )
        except OSError:
            logger.exception("Nao foi possivel gravar o buffer offline")

    def drenar_buffer_offline(self) -> int:
        """
        Reenvia os lotes que ficaram em disco. Devolve quantos subiram.

        Chame ao recuperar conectividade. Os lotes que continuarem falhando são
        reescritos no arquivo, então nada se perde.
        """
        if not self.buffer_offline.exists() or not self.configurado:
            return 0

        linhas = self.buffer_offline.read_text(encoding="utf-8").splitlines()
        if not linhas:
            return 0

        enviados, pendentes = 0, []
        for linha in linhas:
            if not linha.strip():
                continue
            try:
                payload = json.loads(linha)
            except json.JSONDecodeError:
                logger.warning("Linha corrompida no buffer offline, descartada.")
                continue

            if self._post(payload):
                enviados += 1
            else:
                pendentes.append(linha)

        self.buffer_offline.write_text(
            "\n".join(pendentes) + ("\n" if pendentes else ""), encoding="utf-8"
        )
        logger.info("Buffer offline drenado: %d lotes enviados, %d pendentes.",
                    enviados, len(pendentes))
        return enviados

    # -------------------------------------------------------------- utilidades

    def enviar_dataframe(self, df, colunas_extras: Iterable[str] = ()) -> int:
        """
        Envia um DataFrame de features já extraídas — útil para popular a nuvem
        a partir do dataset de treino sem precisar da câmera ligada.

        Espera as colunas `ear`, `mar`, `score_fadiga` e `estado`; as demais
        (`perclos`, `blink_rate`, `pitch`, `yaw`, `roll`) entram se existirem.
        """
        mapa = {
            "perclos": "perclos",
            "blink_rate": "blink_rate",
            "pitch": "head_pitch",
            "yaw": "head_yaw",
            "roll": "head_roll",
            "head_pitch": "head_pitch",
            "head_yaw": "head_yaw",
            "head_roll": "head_roll",
        }

        for _, linha in df.iterrows():
            extras = {
                destino: float(linha[origem])
                for origem, destino in mapa.items()
                if origem in df.columns and linha[origem] == linha[origem]  # descarta NaN
            }
            for coluna in colunas_extras:
                if coluna in df.columns:
                    extras[coluna] = linha[coluna]

            self.registrar_leitura(
                ear=float(linha["ear"]),
                mar=float(linha["mar"]),
                score_fadiga=float(linha["score_fadiga"]),
                estado=str(linha["estado"]),
                **extras,
            )

        self.enviar()
        return self.enviados

    def __enter__(self) -> "DriveGuardClient":
        return self

    def __exit__(self, *_exc) -> None:
        self.enviar()


def _arredondar(valor: float | None, casas: int) -> float | None:
    if valor is None:
        return None
    try:
        if valor != valor:  # NaN
            return None
        return round(float(valor), casas)
    except (TypeError, ValueError):
        return None
