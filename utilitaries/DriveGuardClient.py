"""
Cliente de ingestão do DriveGuard: ponte entre o dispositivo embarcado e a AWS.

Envia as leituras de fadiga para o endpoint `POST /v1/eventos` do API Gateway
provisionado em https://github.com/Drive-Guard/Infra.

Decisões de projeto:

1. **O alerta nunca depende da rede.** A classificação e o alerta acontecem no
   loop da webcam. Este cliente só cuida do que vem depois: mandar a
   telemetria para a nuvem.

2. **O envio não trava a webcam.** Cada lote sobe numa thread própria. Com a
   rede ruim, o POST pode levar dezenas de segundos entre timeout e
   retentativas; no loop principal isso congelaria a imagem e o alerta.

3. **Envio em lote, com buffer em disco.** Um caminhão passa horas sem sinal.
   As leituras se acumulam e sobem juntas; se o POST falhar, o lote vai para
   o buffer offline e é reenviado depois.

4. **Nenhum dado pessoal sai do veículo.** O identificador do motorista e a
   placa são convertidos em SHA-256 antes de qualquer envio, e nenhuma imagem
   é transmitida: só as métricas numéricas.
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
from typing import Any

import requests

logger = logging.getLogger(__name__)

ESTADOS_VALIDOS = {"alerta", "fadiga", "sonolento"}


def hash_identificador(valor: str) -> str:
    """
    Converte um identificador pessoal (CPF, matrícula, placa) em SHA-256.

    É o formato que o schema `silver` espera em `motorista_hash` e
    `placa_hash`. Irreversível: a nuvem nunca recebe o valor original.
    """
    return hashlib.sha256(str(valor).strip().encode("utf-8")).hexdigest()


def agora_utc_iso(ts: datetime | None = None) -> str:
    ts = ts or datetime.now(timezone.utc)
    return ts.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


class DriveGuardClient:
    """Acumula leituras e as envia em lote para a API de ingestão."""

    def __init__(
        self,
        api_url: str,
        api_key: str,
        motorista_id: str,
        device_id: str = "dg-edge-0001",
        veiculo_placa: str | None = None,
        tipo_veiculo: str = "caminhao",
        tamanho_lote: int = 30,
        timeout: int = 10,
        max_tentativas: int = 3,
        buffer_offline: str | Path = "data/temp/buffer_offline.jsonl",
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.device_id = device_id
        self.motorista_hash = hash_identificador(motorista_id)
        self.veiculo_hash = hash_identificador(veiculo_placa) if veiculo_placa else None
        self.tipo_veiculo = tipo_veiculo

        self.tamanho_lote = tamanho_lote
        self.timeout = timeout
        self.max_tentativas = max_tentativas

        self.buffer_offline = Path(buffer_offline)
        self.buffer_offline.parent.mkdir(parents=True, exist_ok=True)

        self._leituras: list[dict[str, Any]] = []
        self._alertas: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._lock_disco = threading.Lock()
        self._threads: list[threading.Thread] = []

        self.enviados = 0
        self.falhas = 0

    @classmethod
    def do_ambiente(cls) -> "DriveGuardClient | None":
        """
        Monta o cliente a partir das variáveis de ambiente (o `.env` do projeto).

        Devolve None quando `DRIVEGUARD_API_URL` não está definida. Assim o
        modo TEST da webcam continua funcionando exatamente como antes para
        quem não configurou a nuvem.
        """
        api_url = os.getenv("DRIVEGUARD_API_URL", "").strip()
        if not api_url:
            return None

        api_key = os.getenv("DRIVEGUARD_API_KEY", "").strip()
        motorista = os.getenv("DRIVEGUARD_MOTORISTA_ID", "").strip()
        if not api_key or not motorista:
            print(
                "Aviso: DRIVEGUARD_API_URL definida, mas faltam DRIVEGUARD_API_KEY "
                "ou DRIVEGUARD_MOTORISTA_ID. Envio para a AWS desligado."
            )
            return None

        return cls(
            api_url=api_url,
            api_key=api_key,
            motorista_id=motorista,
            device_id=os.getenv("DRIVEGUARD_DEVICE_ID", "dg-edge-0001").strip() or "dg-edge-0001",
            veiculo_placa=os.getenv("DRIVEGUARD_PLACA", "").strip() or None,
            tipo_veiculo=os.getenv("DRIVEGUARD_TIPO_VEICULO", "caminhao").strip() or "caminhao",
        )

    # ------------------------------------------------------------------ estado

    def health(self) -> bool:
        """Checa o endpoint /health (não exige API Key)."""
        try:
            r = requests.get(f"{self.api_url}/health", timeout=self.timeout)
            return r.status_code == 200
        except requests.RequestException:
            return False

    # --------------------------------------------------------------- acumulacao

    def registrar_leitura(
        self,
        ear: float,
        score_fadiga: float,
        estado: str,
        mar: float | None = None,
        perclos: float | None = None,
        head_pitch: float | None = None,
        head_yaw: float | None = None,
        head_roll: float | None = None,
        registrado_em: datetime | None = None,
    ) -> None:
        """Adiciona uma leitura ao buffer e dispara o envio ao completar o lote."""
        if estado not in ESTADOS_VALIDOS:
            raise ValueError(f"estado invalido: {estado}")

        leitura = {
            "registrado_em": agora_utc_iso(registrado_em),
            "ear": _arredondar(ear, 4),
            "mar": _arredondar(mar, 4),
            "perclos": _arredondar(perclos, 4),
            "head_pitch": _arredondar(head_pitch, 2),
            "head_yaw": _arredondar(head_yaw, 2),
            "head_roll": _arredondar(head_roll, 2),
            "score_fadiga": _arredondar(score_fadiga, 2),
            "estado": estado,
        }
        leitura = {k: v for k, v in leitura.items() if v is not None}

        with self._lock:
            self._leituras.append(leitura)
            cheio = len(self._leituras) >= self.tamanho_lote

        if cheio:
            self.enviar(assincrono=True)

    def registrar_alerta(
        self,
        tipo: str,
        gravidade: str,
        score_fadiga: float,
        mensagem: str | None = None,
        disparado_em: datetime | None = None,
    ) -> None:
        """Registra, como telemetria, um alerta que já foi emitido no veículo."""
        with self._lock:
            self._alertas.append(
                {
                    "disparado_em": agora_utc_iso(disparado_em),
                    "tipo": tipo,
                    "gravidade": gravidade,
                    "score_fadiga": _arredondar(score_fadiga, 2),
                    "mensagem": mensagem,
                }
            )

    # ------------------------------------------------------------------- envio

    def enviar(self, assincrono: bool = False) -> None:
        """
        Envia o que estiver no buffer.

        `assincrono=True` é o caminho do loop da webcam: o POST roda numa
        thread daemon e o loop segue. No encerramento, use o padrão síncrono
        (ou `finalizar`) para não perder o último lote.
        """
        with self._lock:
            if not self._leituras:
                return
            leituras, self._leituras = self._leituras, []
            alertas, self._alertas = self._alertas, []

        payload = self._montar_payload(leituras, alertas)

        if assincrono:
            t = threading.Thread(target=self._enviar_payload, args=(payload,), daemon=True)
            t.start()
            self._threads = [x for x in self._threads if x.is_alive()] + [t]
        else:
            self._enviar_payload(payload)

    def finalizar(self, espera_max_s: float = 30.0) -> None:
        """Sobe o resto do buffer, espera os envios em andamento e drena o disco."""
        self.enviar()
        limite = time.monotonic() + espera_max_s
        for t in self._threads:
            t.join(timeout=max(0.0, limite - time.monotonic()))
        self.drenar_buffer_offline()

    def _enviar_payload(self, payload: dict[str, Any]) -> None:
        n = len(payload["leituras"])
        if self._post(payload):
            self.enviados += n
        else:
            self.falhas += n
            self._gravar_offline(payload)

    def _montar_payload(self, leituras: list, alertas: list) -> dict[str, Any]:
        return {
            "device_id": self.device_id,
            "schema_version": "1.0",
            "motorista_hash": self.motorista_hash,
            "veiculo_hash": self.veiculo_hash,
            "tipo_veiculo": self.tipo_veiculo,
            "enviado_em": agora_utc_iso(),
            "leituras": leituras,
            "alertas": alertas,
        }

    def _post(self, payload: dict[str, Any]) -> bool:
        headers = {"x-api-key": self.api_key, "Content-Type": "application/json"}
        espera = 1.0

        for tentativa in range(1, self.max_tentativas + 1):
            try:
                r = requests.post(
                    f"{self.api_url}/eventos", headers=headers, json=payload, timeout=self.timeout
                )
                if r.status_code == 202:
                    corpo = r.json()
                    logger.info(
                        "Lote aceito: %s leituras, ingest_id=%s",
                        corpo.get("leituras_recebidas"), corpo.get("ingest_id"),
                    )
                    return True

                # 4xx (exceto 429) é payload malformado: repetir não resolve.
                if 400 <= r.status_code < 500 and r.status_code != 429:
                    logger.error("Lote rejeitado (HTTP %s): %s", r.status_code, r.text[:500])
                    return False

                logger.warning(
                    "API respondeu HTTP %s (tentativa %d/%d)",
                    r.status_code, tentativa, self.max_tentativas,
                )
            except requests.RequestException as exc:
                logger.warning(
                    "Falha de rede (tentativa %d/%d): %s", tentativa, self.max_tentativas, exc
                )

            if tentativa < self.max_tentativas:
                time.sleep(espera)
                espera *= 2

        return False

    # ---------------------------------------------------------- buffer offline

    def _gravar_offline(self, payload: dict[str, Any]) -> None:
        try:
            with self._lock_disco, self.buffer_offline.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
            logger.info("Lote guardado no buffer offline (%d leituras).", len(payload["leituras"]))
        except OSError:
            logger.exception("Nao foi possivel gravar o buffer offline")

    def drenar_buffer_offline(self) -> int:
        """
        Reenvia os lotes que ficaram em disco. Devolve quantos lotes subiram.

        Os que continuarem falhando são reescritos no arquivo: nada se perde.
        """
        with self._lock_disco:
            if not self.buffer_offline.exists():
                return 0
            linhas = [l for l in self.buffer_offline.read_text(encoding="utf-8").splitlines() if l.strip()]

        if not linhas:
            return 0

        subiram, pendentes = 0, []
        for linha in linhas:
            try:
                payload = json.loads(linha)
            except json.JSONDecodeError:
                logger.warning("Linha corrompida no buffer offline, descartada.")
                continue
            if self._post(payload):
                subiram += 1
                self.enviados += len(payload.get("leituras", []))
            else:
                pendentes.append(linha)

        with self._lock_disco:
            self.buffer_offline.write_text(
                "\n".join(pendentes) + ("\n" if pendentes else ""), encoding="utf-8"
            )
        logger.info("Buffer offline: %d lotes enviados, %d pendentes.", subiram, len(pendentes))
        return subiram


def _arredondar(valor: Any, casas: int) -> float | None:
    if valor is None:
        return None
    try:
        v = float(valor)
    except (TypeError, ValueError):
        return None
    if v != v:  # NaN
        return None
    return round(v, casas)
