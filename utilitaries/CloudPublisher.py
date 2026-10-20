"""
Integração entre o modelo de sonolência e a nuvem do DriveGuard.

A classe prevista continua vindo de `pipeline.predict()`, exatamente como no
loop da webcam. O que este módulo acrescenta:

* **três estados**: o modelo treinado com `DRIVEGUARD_CLASSES=3` (padrão) tem
  as classes 0 = alerta, 1 = fadiga e 2 = sonolento, que vão direto para a
  API (`alerta | fadiga | sonolento`). A classe "fadiga" é o meio-termo que o
  rótulo binário descartava: nem todos os sinais de alerta, nem nenhum sinal
  forte de sono (ver `FeatureEngineering.adjust_label_column`);
* **compatibilidade com os .pkl binários** (0 = alerta, 1 = sono, até a v3):
  o estado sai de faixas de P(sono): abaixo de 0,40 alerta, de 0,40 a 0,70
  fadiga, a partir de 0,70 sonolento. Os mesmos cortes valem para o score
  (0-100), então os dois tipos de modelo alimentam o dashboard do mesmo jeito;
* **score de fadiga (0-100)**: `100 x (P(sono) + 0,5 x P(fadiga))` no modelo
  de 3 classes e `100 x P(sono)` no binário. É o que o dashboard usa para
  ordenar motoristas e desenhar a curva de fadiga, então precisa ser contínuo;
* **publicação**: do modelo treinado no S3 e de um CSV de features já
  extraídas na API de ingestão.

Sobre a probabilidade: um VotingClassifier com `voting="hard"` não expõe
`predict_proba`. Nesse caso as probabilidades são a média (ponderada pelos
`weights` do ensemble, se houver) das probabilidades dos membros já
treinados, que é exatamente o que o modo `soft` faria.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from utilitaries.DriveGuardClient import DriveGuardClient

logger = logging.getLogger(__name__)

FEATURES_MODELO = ["ear_behavior", "pitch_behavior", "roll_behavior"]

# Modelo de 3 classes (padrão a partir desta versão).
CLASSE_ALERTA, CLASSE_FADIGA, CLASSE_SONO = 0, 1, 2
ESTADOS = {CLASSE_ALERTA: "alerta", CLASSE_FADIGA: "fadiga", CLASSE_SONO: "sonolento"}
CODIGO_ESTADO = {v: k for k, v in ESTADOS.items()}

# Modelos binários antigos: 0 = alerta, 1 = sono.
CLASSE_SONO_BINARIO = 1
LIMIAR_FADIGA = 0.40
LIMIAR_SONO = 0.70


def _classes(pipeline) -> list[int]:
    return [int(float(c)) for c in pipeline[-1].classes_]


def eh_binario(pipeline) -> bool:
    return len(_classes(pipeline)) <= 2


def probabilidades(pipeline, X: pd.DataFrame) -> np.ndarray:
    """Retorna probabilidades no formato (amostras, classes)."""

    classes = _classes(pipeline)

    if hasattr(pipeline, "predict_proba"):
        P = np.asarray(
            pipeline.predict_proba(X),
            dtype=float
        )
                
        # Uma amostra pode vir representada por um vetor
        # com uma probabilidade para cada classe.
        if P.ndim == 1:
            if len(X) == 1 and P.size == len(classes):
                P = P.reshape(1, -1)
            else:
                raise ValueError(
                    "Formato inesperado de predict_proba: "
                    f"P.shape={P.shape}, "
                    f"X.shape={np.asarray(X).shape}, "
                    f"classes={classes}. "
                    "Esperado: (n_amostras, n_classes)."
                )

        if P.ndim != 2 or P.shape[1] != len(classes):
            raise ValueError(
                "Probabilidades incompatíveis com as classes: "
                f"P.shape={P.shape}, classes={classes}"
            )

        return P

    # Mantém o fallback existente para modelos sem predict_proba.
    final = pipeline[-1]
    membros = getattr(final, "estimators_", None)

    if membros:
        Xt = pipeline[:-1].transform(X)
        probas = np.array([
            m.predict_proba(Xt) for m in membros
        ])
        return np.average(
            probas,
            axis=0,
            weights=getattr(final, "weights", None)
        )

    pred = np.asarray(pipeline.predict(X)).reshape(-1).astype(int)

    return np.array([
        [1.0 if c == p else 0.0 for c in classes]
        for p in pred
    ])


def probabilidade_sono(pipeline, X: pd.DataFrame) -> np.ndarray:
    """P(sono) de cada linha (classe 1 no binário, classe 2 no de 3 classes)."""
    classes = _classes(pipeline)
    alvo = CLASSE_SONO_BINARIO if eh_binario(pipeline) else CLASSE_SONO
    if alvo not in classes:
        raise ValueError(f"O modelo nao tem a classe de sono (classes: {classes})")
    return probabilidades(pipeline, X)[:, classes.index(alvo)]


def estado_por_score(score: float) -> str:
    """Faixas usadas com o modelo binário (e para leitura humana do score)."""
    if score >= LIMIAR_SONO * 100:
        return "sonolento"
    if score >= LIMIAR_FADIGA * 100:
        return "fadiga"
    return "alerta"


def avaliar(pipeline, X: pd.DataFrame) -> list[tuple[str, float]]:
    """
    (estado, score 0-100) de cada linha de X.

    3 classes: o estado é o `predict()` do modelo e o score conta a fadiga
    pela metade. Binário: o score é 100 x P(sono) e o estado sai das faixas.
    """
    classes = _classes(pipeline)
    P = probabilidades(pipeline, X)

    if eh_binario(pipeline):
        scores = P[:, classes.index(CLASSE_SONO_BINARIO)] * 100
        return [(estado_por_score(s), round(float(s), 2)) for s in scores]

    def coluna(c):
        return P[:, classes.index(c)] if c in classes else np.zeros(len(P))

    scores = (coluna(CLASSE_SONO) + 0.5 * coluna(CLASSE_FADIGA)) * 100
    pred = np.asarray(pipeline.predict(X), dtype=float).astype(int)
    return [(ESTADOS[int(p)], round(float(s), 2)) for p, s in zip(pred, scores)]


def estado_da_predicao(pred, binario: bool = False) -> str:
    """Estado da API a partir da classe prevista."""
    c = int(float(pred))
    if binario:
        return "sonolento" if c == CLASSE_SONO_BINARIO else "alerta"
    return ESTADOS[c]


def gravidade(score: float) -> str:
    if score >= 90:
        return "critica"
    if score >= 80:
        return "alta"
    if score >= 60:
        return "media"
    return "baixa"


def tipo_alerta(perclos: float, mar: float, pitch: float, fe) -> str:
    """
    Escolhe o tipo do alerta pela feature que disparou, usando os MESMOS
    limiares que o FeatureEngineering usa para rotular o dataset.
    """
    if perclos is not None and perclos >= fe.PERCLOS_THRESHOLD:
        return "microssono"
    if mar is not None and mar >= fe.YAWN_THRESHOLD:
        return "bocejo_excessivo"
    if pitch is not None and pitch <= fe.HEAD_DROP_THRESHOLD:
        return "cabeca_baixa"
    return "sonolencia"


# ---------------------------------------------------------------------------
# Publicação do modelo no S3
# ---------------------------------------------------------------------------

def publicar_modelo_s3(model_path: str, bucket: str | None = None) -> str | None:
    """
    Envia o .pkl ao bucket de artefatos provisionado pelo repositório Infra,
    numa pasta versionada por data e hora.
    """
    bucket = bucket or os.getenv("DRIVEGUARD_ARTIFACTS_BUCKET", "").strip()
    if not bucket:
        print("DRIVEGUARD_ARTIFACTS_BUCKET nao definido: modelo nao enviado ao S3.")
        return None

    import boto3  # só necessário neste modo

    arquivo = Path(model_path)
    versao = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    chave = f"modelos/{arquivo.stem}/{versao}/{arquivo.name}"
    boto3.client("s3").upload_file(str(arquivo), bucket, chave)

    uri = f"s3://{bucket}/{chave}"
    print(f"Modelo publicado em {uri}")
    return uri


# ---------------------------------------------------------------------------
# Publicação de um CSV de features na API
# ---------------------------------------------------------------------------

def publicar_dataset(
    pipeline,
    csv_path: str,
    fe,
    a_cada: int | None = None,
    lote: int = 500,
) -> int:
    """
    Classifica um CSV gerado pelo modo EXTRACT e envia as leituras à API.

    Serve para alimentar Bronze -> Silver -> Gold com o resultado real do
    modelo sobre o UTA-RLDD, sem precisar da webcam.

    Cada `participant_id` vira um motorista (identificador em hash) e um
    dispositivo próprio. O vídeo não carrega horário de relógio, então
    `registrado_em` é reconstruído a partir de `minute` e `frame` do CSV
    (assumindo 30 fps, o mesmo padrão do FeatureEngineering), ancorado no
    horário atual. A distância temporal real entre as leituras é preservada.

    `a_cada` amostra uma linha a cada N; o padrão (30) equivale a uma leitura
    por segundo de vídeo, a mesma cadência do envio ao vivo.
    """
    api_url = os.getenv("DRIVEGUARD_API_URL", "").strip()
    api_key = os.getenv("DRIVEGUARD_API_KEY", "").strip()
    if not api_url or not api_key:
        print("DRIVEGUARD_API_URL e DRIVEGUARD_API_KEY precisam estar no .env.")
        return 0

    a_cada = a_cada or int(os.getenv("DRIVEGUARD_PUBLISH_STEP", "30"))

    df = pd.read_csv(csv_path, sep=";", encoding="utf-8")
    faltando = [c for c in FEATURES_MODELO + ["participant_id", "minute", "frame"] if c not in df.columns]
    if faltando:
        raise ValueError(f"CSV sem as colunas {faltando}. Gere-o com MODEL_BEHAVIOR=EXTRACT.")

    df = df.dropna(subset=FEATURES_MODELO).sort_values(["participant_id", "minute", "frame"])
    # cumcount em vez de groupby().apply(): no pandas >= 2.2 o apply descarta a
    # coluna do agrupamento, e o participant_id sumiria do resultado.
    df = df[df.groupby("participant_id").cumcount() % a_cada == 0]

    X = df[FEATURES_MODELO]
    avaliacao = avaliar(pipeline, X)
    df["_estado"] = [e for e, _ in avaliacao]
    df["_score"] = [v for _, v in avaliacao]

    base = datetime.now(timezone.utc) - timedelta(minutes=10)
    total = 0

    for pid, grupo in df.groupby("participant_id"):
        cliente = DriveGuardClient(
            api_url=api_url,
            api_key=api_key,
            motorista_id=f"uta-rldd-{pid}",
            device_id=f"dg-dataset-{pid}",
            tamanho_lote=10**9,  # aqui o envio é controlado manualmente
        )
        anterior = "alerta"
        for i, (_, r) in enumerate(grupo.iterrows(), start=1):
            ts = base + timedelta(seconds=float(r["minute"]) * 60 + float(r["frame"]) / 30.0)
            score = float(r["_score"])
            estado = r["_estado"]

            cliente.registrar_leitura(
                ear=r.get("ear_behavior"),
                mar=r.get("mar_behavior", r.get("mar")),
                perclos=r.get("perclos"),
                head_pitch=r.get("pitch_behavior"),
                head_yaw=r.get("yaw_behavior", r.get("yaw")),
                head_roll=r.get("roll_behavior"),
                score_fadiga=score,
                estado=estado,
                registrado_em=ts,
            )

            # Alerta na entrada em sonolento, como no loop da webcam.
            if estado == "sonolento" and anterior != "sonolento":
                cliente.registrar_alerta(
                    tipo=tipo_alerta(r.get("perclos"), r.get("mar_behavior"), r.get("pitch_behavior"), fe),
                    gravidade=gravidade(score),
                    score_fadiga=score,
                    mensagem=f"Participante {pid}: modelo classificou sonolento (score {score:.0f}).",
                    disparado_em=ts,
                )
            anterior = estado

            if i % lote == 0:
                cliente.enviar()
        cliente.enviar()

        print(f"Participante {pid}: {cliente.enviados} leituras enviadas, {cliente.falhas} no buffer offline")
        total += cliente.enviados

    return total
