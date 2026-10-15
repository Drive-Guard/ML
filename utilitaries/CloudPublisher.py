"""
Integração entre o modelo de sonolência e a nuvem do DriveGuard.

Este módulo NÃO altera o modelo nem a forma como ele decide. A classe prevista
continua vindo de `pipeline.predict()`, exatamente como no loop da webcam.
O que ele acrescenta:

* **score de fadiga (0-100)**: `100 x P(sono)`. É o que o dashboard usa para
  ordenar motoristas e desenhar a curva de fadiga, então precisa ser contínuo,
  e não só 0/1;
* **mapeamento para o contrato da API**: o modelo é binário (0 = alerta,
  1 = sono), e a API recebe `alerta | fadiga | sonolento`. O mapeamento é
  direto: 0 -> `alerta`, 1 -> `sonolento`. `fadiga` não é usado porque o modelo
  não tem essa classe, e inventá-la a partir de faixas de probabilidade seria
  criar uma decisão que o modelo não toma;
* **publicação**: do modelo treinado no S3 e de um CSV de features já
  extraídas na API de ingestão.

Sobre a probabilidade: um VotingClassifier com `voting="hard"` não expõe
`predict_proba`. Nesse caso P(sono) é calculada como a média (ponderada pelos
`weights` do ensemble, se houver) das probabilidades dos membros já treinados,
que é exatamente o que o modo `soft` faria. A decisão continua sendo a do
`predict()`; só o score vem dessa média.
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
CLASSE_SONO = 1


def _indice_classe_sono(classes) -> int:
    classes = list(classes)
    for i, c in enumerate(classes):
        if float(c) == float(CLASSE_SONO):
            return i
    raise ValueError(f"O modelo nao tem a classe {CLASSE_SONO} (classes: {classes})")


def probabilidade_sono(pipeline, X: pd.DataFrame) -> np.ndarray:
    """P(sono) para cada linha de X, qualquer que seja o tipo de votação."""
    final = pipeline[-1]

    try:
        proba = pipeline.predict_proba(X)
        return proba[:, _indice_classe_sono(final.classes_)]
    except AttributeError:
        pass

    # voting="hard": média das probabilidades dos membros treinados.
    membros = getattr(final, "estimators_", None)
    if membros:
        Xt = pipeline[:-1].transform(X)
        # Os membros de um VotingClassifier são treinados com os rótulos já
        # codificados (0..n-1, na ordem de final.classes_), então a coluna da
        # classe de sono tem o mesmo índice nos dois.
        idx = _indice_classe_sono(final.classes_)
        probas = np.array([m.predict_proba(Xt)[:, idx] for m in membros])
        return np.average(probas, axis=0, weights=getattr(final, "weights", None))

    # Sem probabilidade disponível: o score fica 0 ou 100.
    return (np.asarray(pipeline.predict(X), dtype=float) == CLASSE_SONO).astype(float)


def estado_da_predicao(pred) -> str:
    return "sonolento" if int(float(pred)) == CLASSE_SONO else "alerta"


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
    df["_pred"] = pipeline.predict(X)
    df["_p_sono"] = probabilidade_sono(pipeline, X)

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
        anterior = 0
        for i, (_, r) in enumerate(grupo.iterrows(), start=1):
            ts = base + timedelta(seconds=float(r["minute"]) * 60 + float(r["frame"]) / 30.0)
            score = round(float(r["_p_sono"]) * 100, 2)
            estado = estado_da_predicao(r["_pred"])

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

            # Alerta na transição alerta -> sono, como no loop da webcam.
            atual = int(float(r["_pred"]))
            if atual == CLASSE_SONO and anterior != CLASSE_SONO:
                cliente.registrar_alerta(
                    tipo=tipo_alerta(r.get("perclos"), r.get("mar_behavior"), r.get("pitch_behavior"), fe),
                    gravidade=gravidade(score),
                    score_fadiga=score,
                    mensagem=f"Participante {pid}: modelo classificou sono (P={score/100:.2f}).",
                    disparado_em=ts,
                )
            anterior = atual

            if i % lote == 0:
                cliente.enviar()
        cliente.enviar()

        print(f"Participante {pid}: {cliente.enviados} leituras enviadas, {cliente.falhas} no buffer offline")
        total += cliente.enviados

    return total
