import numpy as np

from models.FatiguePrediction import FEATURES, FatiguePrediction, gerar_dataset_sintetico, montar_janelas


def test_janelas_tem_features_e_alvo():
    leituras = gerar_dataset_sintetico(turnos=4, semente=1)
    j = montar_janelas(leituras)
    assert set(FEATURES + ["alvo", "turno_id"]).issubset(j.columns)
    assert j["horas_dirigindo"].min() >= 10 / 60 - 1e-9
    assert set(j["alvo"].unique()) <= {0, 1}


def test_treina_preve_e_salva(tmp_path):
    leituras = gerar_dataset_sintetico(turnos=80, semente=7)
    modelo = FatiguePrediction()
    r = modelo.treinar(leituras, str(tmp_path / "preditivo.pkl"))
    assert r.roc_auc > 0.7  # dados sintéticos: só confere que o pipeline aprende
    carregado = FatiguePrediction.carregar(str(tmp_path / "preditivo.pkl"))
    turno = leituras[leituras["turno_id"] == leituras["turno_id"].iloc[0]].head(240)
    p = carregado.prever(turno)
    assert p is None or 0.0 <= p <= 1.0


def test_tendencia_alta_aumenta_a_probabilidade():
    leituras = gerar_dataset_sintetico(turnos=120, semente=3)
    modelo = FatiguePrediction()
    modelo.treinar(leituras)
    j = montar_janelas(leituras)
    baixo = j[(j["score_5min"] < 40) & (j["tendencia_15min"] < 0.2)]
    subindo = j[(j["score_5min"].between(55, 69)) & (j["tendencia_15min"] > 0.5)]
    pb = modelo.modelo.predict_proba(baixo[FEATURES])[:, 1].mean()
    ps = modelo.modelo.predict_proba(subindo[FEATURES])[:, 1].mean()
    assert ps > pb
