"""
Modelo preditivo de base: o motorista vai entrar na zona de alerta nos
próximos minutos?

O classificador de sonolência (Voting / rede neural) diz o estado AGORA. Este
modelo olha a evolução recente do turno e estima a probabilidade de o score
de fadiga passar do limite de alerta (70) dentro do horizonte (15 min), para a
central agir antes do alerta na cabine.

Entrada: leituras em série por turno (CSV com `;`), no formato que o DriveGuard
já guarda em silver.leituras_fadiga:

    turno_id;registrado_em;score_fadiga;perclos;blink_rate

`blink_rate` pode faltar (vem do BlinkDetector; leituras antigas não têm).

Para cada minuto de cada turno, com pelo menos 10 minutos de histórico, monta:

    horas_dirigindo, hora do dia (seno e cosseno: o ritmo circadiano),
    score médio dos últimos 5 e 15 min, tendência (inclinação em pontos/min
    nos últimos 15 min), máximo em 15 min, minutos acima do limite em 30 min,
    PERCLOS e piscadas/min dos últimos 5 min

e o alvo: o score passa do limite em algum momento dos próximos 15 min?
Treina só com minutos em que o motorista ainda está abaixo do limite: o caso
útil é prever a transição, não confirmar quem já está em alerta.

É uma linha de base: HistGradientBoosting (aceita falta de valor, sem escala),
validação separada por turno (o mesmo turno nunca cai em treino e teste).
Enquanto não houver turnos reais suficientes, `gerar_dataset_sintetico` produz
dados com os padrões da literatura (fadiga crescente com as horas ao volante e
pico na madrugada) só para exercitar o pipeline; as métricas sobre esses dados
NÃO valem como resultado do TCC.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

LIMITE_ALERTA = 70.0
HORIZONTE_MIN = 15
HISTORICO_MIN = 10

FEATURES = [
    "horas_dirigindo",
    "hora_sen",
    "hora_cos",
    "score_5min",
    "score_15min",
    "tendencia_15min",
    "max_15min",
    "minutos_acima_30min",
    "perclos_5min",
    "blink_rate_5min",
]


def _inclinacao(valores: np.ndarray) -> float:
    """Inclinação da reta (pontos por minuto) ignorando minutos sem leitura."""
    x = np.arange(len(valores), dtype=float)
    ok = ~np.isnan(valores)
    if ok.sum() < 3:
        return np.nan
    return float(np.polyfit(x[ok], valores[ok], 1)[0])


def montar_janelas(leituras: pd.DataFrame, horizonte_min: int = HORIZONTE_MIN, com_alvo: bool = True) -> pd.DataFrame:
    """Uma linha por minuto de turno, com as features e (opcionalmente) o alvo."""
    df = leituras.copy()
    df["registrado_em"] = pd.to_datetime(df["registrado_em"], utc=True)
    if "blink_rate" not in df.columns:
        df["blink_rate"] = np.nan
    linhas = []

    for turno, g in df.sort_values("registrado_em").groupby("turno_id"):
        por_minuto = (
            g.set_index("registrado_em")[["score_fadiga", "perclos", "blink_rate"]]
            .resample("1min")
            .agg({"score_fadiga": ["mean", "max"], "perclos": "mean", "blink_rate": "mean"})
        )
        por_minuto.columns = ["score", "score_max", "perclos", "blink_rate"]
        inicio = por_minuto.index[0]
        score = por_minuto["score"].to_numpy()
        score_max = por_minuto["score_max"].to_numpy()
        perclos = por_minuto["perclos"].to_numpy()
        blink = por_minuto["blink_rate"].to_numpy()
        n = len(por_minuto)

        ultimo = n - horizonte_min if com_alvo else n
        for t in range(HISTORICO_MIN, ultimo):
            if np.isnan(score[t]):
                continue
            instante = por_minuto.index[t]
            hora_local = (instante.tz_convert("America/Sao_Paulo").hour + instante.minute / 60)
            janela5 = slice(t - 4, t + 1)
            janela15 = slice(max(0, t - 14), t + 1)
            janela30 = slice(max(0, t - 29), t + 1)
            linha = {
                "turno_id": turno,
                "instante": instante,
                "horas_dirigindo": (instante - inicio).total_seconds() / 3600,
                "hora_sen": math.sin(2 * math.pi * hora_local / 24),
                "hora_cos": math.cos(2 * math.pi * hora_local / 24),
                "score_5min": np.nanmean(score[janela5]),
                "score_15min": np.nanmean(score[janela15]),
                "tendencia_15min": _inclinacao(score[janela15]),
                "max_15min": np.nanmax(score_max[janela15]),
                "minutos_acima_30min": int(np.nansum(score[janela30] >= LIMITE_ALERTA)),
                "perclos_5min": np.nanmean(perclos[janela5]) if not np.all(np.isnan(perclos[janela5])) else np.nan,
                "blink_rate_5min": np.nanmean(blink[janela5]) if not np.all(np.isnan(blink[janela5])) else np.nan,
            }
            if com_alvo:
                futuro = score_max[t + 1 : t + 1 + horizonte_min]
                if np.all(np.isnan(futuro)):
                    continue
                linha["alvo"] = int(np.nanmax(futuro) >= LIMITE_ALERTA)
            linhas.append(linha)

    return pd.DataFrame(linhas)


@dataclass
class ResultadoTreino:
    roc_auc: float
    pr_auc: float
    taxa_positivos: float
    linhas_treino: int
    linhas_teste: int
    turnos: int


class FatiguePrediction:
    def __init__(self, horizonte_min: int = HORIZONTE_MIN, limiar_decisao: float = 0.5) -> None:
        self.model_name = "fatigue_prediction"
        self.horizonte_min = horizonte_min
        self.limiar_decisao = limiar_decisao
        self.modelo: HistGradientBoostingClassifier | None = None

    def criar_modelo(self) -> HistGradientBoostingClassifier:
        return HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=42,
        )

    def treinar(self, leituras: pd.DataFrame, caminho_modelo: str | None = None) -> ResultadoTreino:
        janelas = montar_janelas(leituras, self.horizonte_min)
        # Só minutos ainda abaixo do limite: prever a transição.
        janelas = janelas[janelas["score_5min"] < LIMITE_ALERTA].reset_index(drop=True)
        if janelas["alvo"].nunique() < 2:
            raise ValueError("O conjunto não tem exemplos das duas classes; faltam turnos com e sem fadiga.")

        X, y, grupos = janelas[FEATURES], janelas["alvo"], janelas["turno_id"]
        treino, teste = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42).split(X, y, grupos))
        self.modelo = self.criar_modelo().fit(X.iloc[treino], y.iloc[treino])

        p = self.modelo.predict_proba(X.iloc[teste])[:, 1]
        y_teste = y.iloc[teste]
        print("=" * 67)
        print(f"Horizonte: {self.horizonte_min} min | limite de alerta: {LIMITE_ALERTA}")
        print(classification_report(y_teste, (p >= self.limiar_decisao).astype(int), digits=3))
        resultado = ResultadoTreino(
            roc_auc=round(float(roc_auc_score(y_teste, p)), 4),
            pr_auc=round(float(average_precision_score(y_teste, p)), 4),
            taxa_positivos=round(float(y.mean()), 4),
            linhas_treino=len(treino),
            linhas_teste=len(teste),
            turnos=int(grupos.nunique()),
        )
        print(resultado)
        print("=" * 67)
        if caminho_modelo:
            self.salvar(caminho_modelo)
        return resultado

    def prever(self, leituras_do_turno: pd.DataFrame) -> float | None:
        """Probabilidade de passar do limite no horizonte, a partir das leituras do turno até agora."""
        if self.modelo is None:
            raise RuntimeError("Modelo não treinado nem carregado.")
        janelas = montar_janelas(leituras_do_turno, self.horizonte_min, com_alvo=False)
        if janelas.empty:
            return None
        return float(self.modelo.predict_proba(janelas[FEATURES].tail(1))[:, 1][0])

    def salvar(self, caminho: str) -> None:
        joblib.dump(
            {"modelo": self.modelo, "features": FEATURES, "horizonte_min": self.horizonte_min,
             "limite_alerta": LIMITE_ALERTA, "limiar_decisao": self.limiar_decisao},
            caminho,
        )
        print(f"Modelo preditivo salvo em {caminho}")

    @classmethod
    def carregar(cls, caminho: str) -> "FatiguePrediction":
        # joblib desserializa com pickle: só carregue modelos gerados pela
        # própria equipe (mesmo cuidado do Utils.load_model).
        dados = joblib.load(caminho)
        inst = cls(horizonte_min=dados["horizonte_min"], limiar_decisao=dados["limiar_decisao"])
        inst.modelo = dados["modelo"]
        return inst


def gerar_dataset_sintetico(turnos: int = 240, semente: int = 42) -> pd.DataFrame:
    """
    Leituras sintéticas (uma a cada 30 s) só para exercitar o pipeline.

    Cada turno tem um início (dia ou noite), uma duração de 4 a 11 h e um
    motorista com resistência própria. O score sobe com as horas ao volante,
    ganha um pico na madrugada (2h às 6h) e tem ruído; pausas derrubam o score.
    """
    rng = np.random.default_rng(semente)
    linhas = []
    base = pd.Timestamp("2026-09-01T00:00:00Z")
    for turno in range(turnos):
        inicio = base + pd.Timedelta(days=int(rng.integers(0, 40)), hours=float(rng.uniform(0, 24)))
        duracao_h = float(rng.uniform(4, 11))
        resistencia = float(rng.normal(1.0, 0.25))
        pausa_em = float(rng.uniform(2, duracao_h)) if rng.random() < 0.4 else None
        passos = int(duracao_h * 120)
        for k in range(passos):
            instante = inicio + pd.Timedelta(seconds=30 * k)
            horas = k / 120
            hora_local = (instante.tz_convert("America/Sao_Paulo").hour + instante.minute / 60)
            efetivas = horas if pausa_em is None or horas < pausa_em else horas - pausa_em * 0.7
            madrugada = 16 * math.exp(-((hora_local - 4) ** 2) / 4) if hora_local < 9 else 0
            score = 12 + efetivas * 7.0 / max(0.4, resistencia) + madrugada + rng.normal(0, 5)
            score = float(min(99, max(1, score)))
            linhas.append({
                "turno_id": f"sint-{turno:04d}",
                "registrado_em": instante.isoformat(),
                "score_fadiga": round(score, 2),
                "perclos": round(min(0.6, max(0, 0.03 + score / 220 + rng.normal(0, 0.02))), 4),
                "blink_rate": round(max(5, 16 + score / 8 + rng.normal(0, 2)), 2),
            })
    return pd.DataFrame(linhas)
