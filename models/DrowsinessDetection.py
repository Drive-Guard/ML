"""
Detecção de sonolência — ensemble por votação (VotingClassifier).

Substitui os modelos isolados (Random Forest e AdaBoost avaliados
separadamente) por um único ensemble de votação soft, que é o modelo de
produção embarcado no veículo.

Por que votação, e não um modelo só
-----------------------------------
Os estimadores erram de formas diferentes, e é justamente isso que o ensemble
aproveita:

* **Random Forest** — bagging, variância baixa, robusto a outlier de landmark
  mal detectado num frame isolado.
* **Gradient Boosting** — boosting, captura a interação entre features
  (ex.: PERCLOS alto *junto com* pitch negativo) que a floresta dilui.
* **Regressão Logística** — modelo linear calibrado; segura os dois anteriores
  quando eles concordam por overfitting numa região pouco povoada do espaço.

A votação **soft** (média das probabilidades) é usada em vez da hard porque o
produto não precisa só da classe: precisa de um **score contínuo de 0 a 100**
para emitir alertas progressivos *antes* do estado crítico. Esse score sai
direto das probabilidades do ensemble (ver `calcular_score_fadiga`), o que o
torna explicável — diferente de uma fórmula de pesos arbitrários.

Conexão com a AWS
-----------------
Depois de treinar, o modelo é serializado e enviado ao bucket de artefatos
provisionado pelo repositório Infra, e as features do dataset são publicadas na
API de ingestão (`POST /v1/eventos`), alimentando Bronze → Silver → Gold e,
por consequência, o dashboard.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from interfaces.BaseModelo import BaseModelo
from utilitaries.DriveGuardClient import DriveGuardClient
from utilitaries.FeatureEngineering import FeatureEngineering
from utilitaries.Utils import Utils

logger = logging.getLogger(__name__)

FEATURES = ["ear", "mar", "pitch", "yaw", "roll"]

# Rótulo numérico -> estado esperado pelo schema silver da AWS.
MAPA_ESTADOS = {0: "alerta", 1: "fadiga", 2: "sonolento"}

# Peso de risco por classe. O score é a esperança do risco sob a distribuição
# de probabilidade do ensemble: alerta não soma, fadiga soma metade,
# sonolento soma tudo.
PESO_RISCO = {"alerta": 0.0, "fadiga": 0.5, "sonolento": 1.0}


class DrowsinessDetection(BaseModelo):
    """Ensemble de votação para classificar o estado de fadiga do motorista."""

    def __init__(
        self,
        caminho_features: str = "data/features_data_trusted.csv",
        caminho_modelo: str = "data/modelo_voting.joblib",
        random_state: int = 42,
    ) -> None:
        super().__init__()
        self.utils = Utils()
        self.caminho_features = Path(caminho_features)
        self.caminho_modelo = Path(caminho_modelo)
        self.random_state = random_state
        self.pipeline: ImbPipeline | None = None
        self.metricas: dict = {}

    # ------------------------------------------------------------------ dados

    def carregar_dados(self) -> pd.DataFrame:
        if not self.caminho_features.exists():
            logger.info("Features ausentes, extraindo de data/train...")
            FeatureEngineering().extract_features_from_image("data/train")

        df = self.utils.create_dataframe(str(self.caminho_features))
        if df is None or df.empty:
            raise RuntimeError(
                f"Nao foi possivel carregar features de {self.caminho_features}"
            )
        return df

    def split_data(self, df: pd.DataFrame):
        return self.utils.create_train_test_split(
            df, FEATURES, "label", test_size=0.3, random_state=self.random_state
        )

    # --------------------------------------------------------------- pipeline

    def create_pipeline(self) -> ImbPipeline:
        """
        Monta o pipeline completo: escala → SMOTE → votação.

        SMOTE fica **dentro** do pipeline de propósito. Aplicá-lo antes do
        split, como no código anterior, sintetiza amostras a partir de exemplos
        que depois caem no teste — o modelo é avaliado com dados derivados do
        que ele já viu, e a métrica sai otimista. Dentro do pipeline, o imblearn
        garante que o SMOTE só rode na parte de treino de cada fold.
        """
        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=self.random_state,
        )

        gb = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.08,
            max_depth=3,
            subsample=0.9,
            random_state=self.random_state,
        )

        lr = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=self.random_state,
        )

        # Pesos: as árvores dominam porque as features de fadiga têm fronteiras
        # não lineares (EAR cai pela metade, PERCLOS sobe ~7x entre alerta e
        # sonolento); a logística entra com peso menor, como regularizadora.
        voting = VotingClassifier(
            estimators=[("rf", rf), ("gb", gb), ("lr", lr)],
            voting="soft",
            weights=[2, 2, 1],
            n_jobs=-1,
        )

        return ImbPipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("smote", SMOTE(random_state=self.random_state, k_neighbors=5)),
                ("voting", voting),
            ]
        )

    # ---------------------------------------------------------------- treino

    def train_model(self, validacao_cruzada: bool = True) -> dict:
        df = self.carregar_dados()
        X_train, X_test, y_train, y_test = self.split_data(df)

        self.pipeline = self.create_pipeline()

        if validacao_cruzada:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            scores = cross_val_score(
                self.pipeline, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1
            )
            logger.info(
                "F1-macro em validacao cruzada: %.4f (+/- %.4f)", scores.mean(), scores.std()
            )
            self.metricas["cv_f1_macro_media"] = float(scores.mean())
            self.metricas["cv_f1_macro_desvio"] = float(scores.std())

        self.pipeline.fit(X_train, y_train)
        y_pred = self.pipeline.predict(X_test)

        self.utils.create_model_evaluation_report(y_test, y_pred, "Voting (RF + GB + LR)")

        self.metricas.update(
            {
                "modelo": "VotingClassifier(soft, RF+GB+LR)",
                "treinado_em": datetime.now(timezone.utc).isoformat(),
                "amostras_treino": int(len(X_train)),
                "amostras_teste": int(len(X_test)),
                "classes": sorted(int(c) for c in pd.unique(df["label"])),
                "features": FEATURES,
                "acuracia_teste": float((y_pred == y_test).mean()),
            }
        )

        # Contribuição individual de cada membro, para o comparativo do TCC.
        self.metricas["acuracia_por_estimador"] = self._avaliar_membros(
            X_train, X_test, y_train, y_test
        )

        return self.metricas

    def _avaliar_membros(self, X_train, X_test, y_train, y_test) -> dict:
        """Treina cada estimador isolado, para mostrar o ganho do ensemble."""
        resultados = {}
        voting: VotingClassifier = self.pipeline.named_steps["voting"]

        for nome, estimador in voting.estimators:
            pipe = ImbPipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("smote", SMOTE(random_state=self.random_state, k_neighbors=5)),
                    (nome, estimador),
                ]
            )
            pipe.fit(X_train, y_train)
            acuracia = float((pipe.predict(X_test) == y_test).mean())
            resultados[nome] = acuracia
            logger.info("Acuracia isolada de %s: %.4f", nome, acuracia)

        resultados["voting"] = self.metricas["acuracia_teste"]
        return resultados

    def test_model(self, plotar_matriz: bool = True) -> None:
        if self.pipeline is None:
            raise RuntimeError("Treine o modelo antes de testar.")

        df = self.carregar_dados()
        _, X_test, _, y_test = self.split_data(df)
        y_pred = self.pipeline.predict(X_test)

        self.utils.create_model_evaluation_report(y_test, y_pred, "Voting (RF + GB + LR)")
        if plotar_matriz:
            self.utils.create_confusion_matrix(
                y_test, y_pred, self.pipeline.named_steps["voting"], "Voting (RF + GB + LR)"
            )

    # -------------------------------------------------------------- inferencia

    def prever(self, features: dict | pd.DataFrame) -> dict:
        """
        Classifica uma leitura e devolve estado, score e probabilidades.

        É este o método que o loop de captura do veículo chama a cada janela.
        """
        if self.pipeline is None:
            self.carregar_modelo()

        if isinstance(features, dict):
            entrada = pd.DataFrame([{f: features.get(f, 0.0) for f in FEATURES}])
        else:
            entrada = features[FEATURES]

        probabilidades = self.pipeline.predict_proba(entrada)[0]
        classes = list(self.pipeline.named_steps["voting"].classes_)

        distribuicao = {
            MAPA_ESTADOS.get(int(c), str(c)): float(p)
            for c, p in zip(classes, probabilidades)
        }

        score = self.calcular_score_fadiga(distribuicao)
        estado = max(distribuicao, key=distribuicao.get)

        return {
            "estado": estado,
            "score_fadiga": score,
            "probabilidades": distribuicao,
        }

    @staticmethod
    def calcular_score_fadiga(distribuicao: dict[str, float]) -> float:
        """
        Score de 0 a 100 derivado das probabilidades do ensemble.

        `score = 100 * Σ P(classe) * peso_risco(classe)`

        Resolve a pendência apontada na dissertação (seção 8, "definir a
        fórmula do score"): em vez de somar features com pesos escolhidos à
        mão, o score é a **esperança do risco** sob a distribuição que o
        ensemble produz. Isso traz três propriedades úteis:

        * é contínuo, então o alerta pode ser progressivo em vez de binário;
        * é monotônico — mais massa de probabilidade em `sonolento` sempre
          aumenta o score;
        * acompanha a incerteza: um caso ambíguo entre `alerta` e `sonolento`
          cai no meio da escala, e não num extremo.

        A escala casa com a do dashboard (`riskFromScore` em `mockData.ts`):
        <35 baixo · <60 médio · <80 alto · >=80 crítico.
        """
        score = sum(
            probabilidade * PESO_RISCO.get(estado, 0.0)
            for estado, probabilidade in distribuicao.items()
        )
        return round(float(np.clip(score * 100.0, 0.0, 100.0)), 2)

    @staticmethod
    def classificar_gravidade(score: float) -> str:
        if score >= 90:
            return "critica"
        if score >= 80:
            return "alta"
        if score >= 60:
            return "media"
        return "baixa"

    # ------------------------------------------------------------ persistencia

    def salvar_modelo(self) -> Path:
        import joblib

        if self.pipeline is None:
            raise RuntimeError("Nao ha modelo treinado para salvar.")

        self.caminho_modelo.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"pipeline": self.pipeline, "features": FEATURES, "metricas": self.metricas},
            self.caminho_modelo,
        )

        caminho_metricas = self.caminho_modelo.with_suffix(".metricas.json")
        caminho_metricas.write_text(
            json.dumps(self.metricas, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        logger.info("Modelo salvo em %s", self.caminho_modelo)
        return self.caminho_modelo

    def carregar_modelo(self) -> None:
        # joblib desserializa pickle, o que executa código arbitrário: só
        # carregue artefato gerado por `salvar_modelo` ou baixado do bucket de
        # artefatos do próprio projeto, nunca de origem externa. Não há formato
        # alternativo — um Pipeline scikit-learn treinado não serializa em JSON.
        import joblib

        if not self.caminho_modelo.exists():
            raise FileNotFoundError(
                f"Modelo nao encontrado em {self.caminho_modelo}. Rode o treino antes."
            )

        artefato = joblib.load(self.caminho_modelo)
        self.pipeline = artefato["pipeline"]
        self.metricas = artefato.get("metricas", {})
        logger.info("Modelo carregado de %s", self.caminho_modelo)

    # -------------------------------------------------------------------- AWS

    def publicar_modelo_no_s3(self, bucket: str | None = None, prefixo: str = "modelos/") -> str | None:
        """
        Envia o modelo treinado para o bucket de artefatos provisionado no repo
        Infra. O SageMaker e qualquer redeploy do edge leem daqui.
        """
        bucket = bucket or os.environ.get("DRIVEGUARD_ARTIFACTS_BUCKET", "")
        if not bucket:
            logger.warning(
                "DRIVEGUARD_ARTIFACTS_BUCKET nao definido: modelo nao publicado no S3."
            )
            return None

        try:
            import boto3
        except ImportError:
            logger.warning("boto3 nao instalado: modelo nao publicado no S3.")
            return None

        if not self.caminho_modelo.exists():
            self.salvar_modelo()

        versao = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        chave = f"{prefixo}voting/{versao}/{self.caminho_modelo.name}"

        s3 = boto3.client("s3")
        s3.upload_file(str(self.caminho_modelo), bucket, chave)
        s3.put_object(
            Bucket=bucket,
            Key=f"{prefixo}voting/{versao}/metricas.json",
            Body=json.dumps(self.metricas, indent=2, ensure_ascii=False).encode("utf-8"),
            ContentType="application/json",
        )

        uri = f"s3://{bucket}/{chave}"
        logger.info("Modelo publicado em %s", uri)
        return uri

    def publicar_leituras_na_api(self, limite: int | None = None) -> int:
        """
        Classifica o dataset de features e envia as leituras à API de ingestão.

        É o caminho mais rápido para alimentar Bronze → Silver → Gold com dados
        reais do modelo, sem precisar da câmera ligada — útil para validar a
        esteira inteira e para popular o dashboard antes da banca.
        """
        if self.pipeline is None:
            self.carregar_modelo()

        df = self.carregar_dados()
        if limite:
            df = df.head(limite)

        cliente = DriveGuardClient()
        if not cliente.configurado:
            logger.warning(
                "Cliente nao configurado (DRIVEGUARD_API_URL / API_KEY / MOTORISTA_ID). "
                "As leituras vao apenas para o buffer offline."
            )

        probabilidades = self.pipeline.predict_proba(df[FEATURES])
        classes = list(self.pipeline.named_steps["voting"].classes_)

        for (_, linha), probs in zip(df.iterrows(), probabilidades):
            distribuicao = {
                MAPA_ESTADOS.get(int(c), str(c)): float(p) for c, p in zip(classes, probs)
            }
            score = self.calcular_score_fadiga(distribuicao)
            estado = max(distribuicao, key=distribuicao.get)

            cliente.registrar_leitura(
                ear=float(linha["ear"]),
                mar=float(linha["mar"]),
                head_pitch=float(linha["pitch"]),
                head_yaw=float(linha["yaw"]),
                head_roll=float(linha["roll"]),
                score_fadiga=score,
                estado=estado,
            )

            if score >= 75:
                cliente.registrar_alerta(
                    tipo="sonolencia",
                    gravidade=self.classificar_gravidade(score),
                    score_fadiga=score,
                    mensagem=f"Ensemble classificou como {estado} (score {score}).",
                )

        cliente.enviar(forcar=True)
        logger.info("Leituras enviadas a API: %d (falhas: %d)", cliente.enviados, cliente.falhas)
        return cliente.enviados
