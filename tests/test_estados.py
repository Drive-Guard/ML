"""Três estados (alerta / fadiga / sonolento) e compatibilidade com o binário."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utilitaries import CloudPublisher as cp

COLUNAS = cp.FEATURES_MODELO


def _dados(n_classes: int, n: int = 600):
    """EAR cai e a cabeça desce conforme a classe sobe (0 alerta ... 2 sono)."""
    rng = np.random.default_rng(7)
    y = rng.integers(0, n_classes, n)
    passo = 1 if n_classes == 3 else 2
    X = pd.DataFrame({
        "ear_behavior": 0.31 - 0.07 * y * passo + rng.normal(0, 0.01, n),
        "pitch_behavior": -9 * y * passo + rng.normal(0, 1.5, n),
        "roll_behavior": rng.normal(0, 2, n),
    })
    return X[COLUNAS], y


def _pipeline(modelo, n_classes):
    X, y = _dados(n_classes)
    return Pipeline([("scaler", StandardScaler()), ("modelo", modelo)]).fit(X, y)


@pytest.fixture(scope="module")
def tres_classes():
    return _pipeline(RandomForestClassifier(n_estimators=30, random_state=0), 3)


@pytest.fixture(scope="module")
def binario():
    return _pipeline(LogisticRegression(), 2)


def _amostra(classe: int, n_classes: int = 3) -> pd.DataFrame:
    passo = 1 if n_classes == 3 else 2
    return pd.DataFrame([{"ear_behavior": 0.31 - 0.07 * classe * passo,
                          "pitch_behavior": -9 * classe * passo,
                          "roll_behavior": 0.0}])[COLUNAS]


def test_modelo_de_tres_classes_devolve_os_tres_estados(tres_classes):
    assert not cp.eh_binario(tres_classes)
    estados = [cp.avaliar(tres_classes, _amostra(c))[0][0] for c in (0, 1, 2)]
    assert estados == ["alerta", "fadiga", "sonolento"]


def test_score_cresce_com_o_estado(tres_classes):
    scores = [cp.avaliar(tres_classes, _amostra(c))[0][1] for c in (0, 1, 2)]
    assert scores[0] < 40 <= scores[1] < 70 <= scores[2]
    assert all(0 <= s <= 100 for s in scores)


def test_probabilidade_de_sono_usa_a_classe_2(tres_classes):
    p = cp.probabilidade_sono(tres_classes, _amostra(2))
    assert p[0] > 0.7


def test_binario_usa_faixas_de_probabilidade(binario):
    assert cp.eh_binario(binario)
    estado_alerta, score_alerta = cp.avaliar(binario, _amostra(0, 2))[0]
    estado_sono, score_sono = cp.avaliar(binario, _amostra(1, 2))[0]
    assert (estado_alerta, estado_sono) == ("alerta", "sonolento")
    assert score_alerta < 40 and score_sono >= 70


@pytest.mark.parametrize("score,estado", [(0, "alerta"), (39.9, "alerta"), (40, "fadiga"),
                                          (69.9, "fadiga"), (70, "sonolento"), (100, "sonolento")])
def test_faixas_do_score(score, estado):
    assert cp.estado_por_score(score) == estado


def test_votacao_hard_tambem_tem_tres_probabilidades():
    hard = _pipeline(VotingClassifier([("rf", RandomForestClassifier(n_estimators=20, random_state=0)),
                                       ("lr", LogisticRegression(max_iter=500))], voting="hard"), 3)
    P = cp.probabilidades(hard, _amostra(1))
    assert P.shape == (1, 3)
    assert np.isclose(P.sum(), 1.0)
    assert cp.avaliar(hard, _amostra(2))[0][0] == "sonolento"


def test_codigos_dos_estados():
    assert cp.CODIGO_ESTADO == {"alerta": 0, "fadiga": 1, "sonolento": 2}
    assert cp.estado_da_predicao(1) == "fadiga"
    assert cp.estado_da_predicao(1, binario=True) == "sonolento"


def test_rotulo_de_tres_classes_mantem_o_meio_termo(monkeypatch):
    # O rótulo não usa câmera nem gráfico: stub do que não estiver instalado.
    import importlib
    import sys
    from unittest.mock import MagicMock
    for nome in ("cv2", "mediapipe", "mediapipe.tasks", "mediapipe.tasks.python",
                 "mediapipe.tasks.python.vision", "matplotlib", "matplotlib.pyplot"):
        try:
            importlib.import_module(nome)
        except ImportError:
            monkeypatch.setitem(sys.modules, nome, MagicMock())
    fe_mod = importlib.import_module("utilitaries.FeatureEngineering")
    fe = fe_mod.FeatureEngineering.__new__(fe_mod.FeatureEngineering)
    fe.ALERT_PERCLOS_THRESHOLD, fe.PERCLOS_THRESHOLD = 0.02, 0.15
    fe.CLOSED_MOUTH_THRESHOLD, fe.YAWN_THRESHOLD = 0.10, 0.60
    fe.HEAD_DROP_THRESHOLD = -15

    df = pd.DataFrame({
        "participant_id": [1, 2, 3],
        # alerta, meio-termo (PERCLOS entre os limiares), sono (PERCLOS alto)
        "perclos": [0.01, 0.08, 0.30],
        "ear": [0.31, 0.25, 0.15],
        "pitch": [0.0, -8.0, -20.0],
        "yaw": [0.0, 0.0, 0.0],
        "roll": [0.0, 0.0, 0.0],
        "mar": [0.05, 0.05, 0.05],
    })

    fe.NUM_CLASSES = 3
    assert fe.adjust_label_column(df.copy())["label"].tolist() == [0, 1, 2]

    fe.NUM_CLASSES = 2
    assert fe.adjust_label_column(df.copy())["label"].tolist() == [0, 1]
