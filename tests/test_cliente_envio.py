"""Cliente de ingestão: lote configurável, alerta sem esperar o lote e câmera padrão."""

import pytest

from utilitaries import DriveGuardClient as dg


@pytest.fixture
def cliente(tmp_path, monkeypatch):
    enviados = []
    monkeypatch.setattr(dg.DriveGuardClient, "_post", lambda self, payload: enviados.append(payload) or True)
    c = dg.DriveGuardClient("https://api.teste/v1", "chave", "DRV-2026", device_id="dg-demo-caetano",
                            tamanho_lote=5, buffer_offline=tmp_path / "buffer.jsonl")
    c.enviados_payloads = enviados
    return c


def esperar(cliente):
    """Espera os envios em andamento (sem mandar o que sobrou no buffer, como o finalizar)."""
    for t in list(cliente._threads):
        t.join(timeout=5)


def ler(cliente, n=1, estado="alerta"):
    for _ in range(n):
        cliente.registrar_leitura(ear=0.3, score_fadiga=30, estado=estado)


def test_lote_configuravel(cliente):
    ler(cliente, 4)
    esperar(cliente)
    assert cliente.enviados_payloads == []  # ainda não completou 5
    ler(cliente, 1)
    esperar(cliente)
    assert len(cliente.enviados_payloads) == 1 and len(cliente.enviados_payloads[0]["leituras"]) == 5


def test_alerta_sobe_sem_esperar_o_lote(cliente):
    ler(cliente, 2, estado="sonolento")
    cliente.registrar_alerta(tipo="sonolencia", gravidade="critica", score_fadiga=92)
    esperar(cliente)
    assert len(cliente.enviados_payloads) == 1
    assert cliente.enviados_payloads[0]["alertas"][0]["tipo"] == "sonolencia"
    # Alerta com o buffer vazio: vai junto com a próxima leitura.
    cliente.registrar_alerta(tipo="sonolencia", gravidade="alta", score_fadiga=80)
    esperar(cliente)
    assert len(cliente.enviados_payloads) == 1
    ler(cliente, 1, estado="sonolento")
    esperar(cliente)
    assert len(cliente.enviados_payloads) == 2 and len(cliente.enviados_payloads[1]["alertas"]) == 1


def test_variaveis_do_ambiente(monkeypatch):
    monkeypatch.setenv("DRIVEGUARD_API_URL", "https://api.teste/v1")
    monkeypatch.setenv("DRIVEGUARD_API_KEY", "chave")
    monkeypatch.setenv("DRIVEGUARD_MOTORISTA_ID", "DRV-2026")
    monkeypatch.setenv("DRIVEGUARD_TAMANHO_LOTE", "5")
    monkeypatch.delenv("DRIVEGUARD_DEVICE_ID", raising=False)
    monkeypatch.setattr(dg.socket, "gethostname", lambda: "Notebook_Caetano")
    c = dg.DriveGuardClient.do_ambiente()
    assert c.tamanho_lote == 5
    # Sem DRIVEGUARD_DEVICE_ID, a câmera é a do computador, nunca uma simulada.
    assert c.device_id == "dg-notebook-caetano"
    monkeypatch.setenv("DRIVEGUARD_TAMANHO_LOTE", "abc")
    assert dg.DriveGuardClient.do_ambiente().tamanho_lote == 30
