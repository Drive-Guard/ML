"""Envio do modelo pela API (link pré-assinado), sem credencial da AWS no computador."""

import types

import pytest

from utilitaries import CloudPublisher as cp


class Resposta:
    def __init__(self, status, dados=None, texto=""):
        self.status_code = status
        self._dados = dados or {}
        self.text = texto

    def json(self):
        return self._dados

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


@pytest.fixture
def modelo(tmp_path, monkeypatch):
    monkeypatch.setenv("DRIVEGUARD_API_URL", "https://api.teste/v1")
    monkeypatch.setenv("DRIVEGUARD_API_KEY", "chave")
    monkeypatch.setattr(cp.time, "sleep", lambda _s: None)
    arquivo = tmp_path / "pipeline_voting.pkl"
    arquivo.write_bytes(b"modelo" * 1000)
    return arquivo


def instalar_requests(monkeypatch, envios):
    pedidos = []

    def post(url, json, headers, timeout):
        pedidos.append((url, json, headers))
        n = len(pedidos)
        return Resposta(201, {"url": f"https://s3/link-{n}", "cabecalhos": {"Content-Type": "application/octet-stream"}, "uri": "s3://artefatos/modelos/x.pkl"})

    def put(url, data, headers, timeout):
        data.read()
        return Resposta(envios.pop(0))

    monkeypatch.setitem(__import__("sys").modules, "requests", types.SimpleNamespace(post=post, put=put, RequestException=Exception))
    return pedidos


def test_link_vencido_pede_outro_e_envia(modelo, monkeypatch):
    pedidos = instalar_requests(monkeypatch, [403, 200])
    assert cp.publicar_modelo_s3(str(modelo)) == "s3://artefatos/modelos/x.pkl"
    assert len(pedidos) == 2  # um link novo por tentativa
    url, corpo, cabecalhos = pedidos[0]
    assert url == "https://api.teste/v1/modelos"
    assert corpo == {"arquivo": "pipeline_voting.pkl", "tamanho": 6000}
    assert cabecalhos == {"x-api-key": "chave"}


def test_desiste_depois_das_tentativas_sem_perder_o_arquivo(modelo, monkeypatch):
    instalar_requests(monkeypatch, [500, 500, 500, 500])
    assert cp.publicar_modelo_s3(str(modelo)) is None
    assert modelo.exists()
