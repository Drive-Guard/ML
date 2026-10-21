"""IDs do .env normalizados do mesmo jeito que a dashboard (casar o cadastro com a telemetria)."""

import hashlib

from utilitaries.DriveGuardClient import (
    DriveGuardClient,
    hash_motorista,
    normalizar_dispositivo,
    normalizar_id,
    normalizar_tipo_veiculo,
)


def sha(texto: str) -> str:
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def test_mesmo_motorista_escrito_de_jeitos_diferentes():
    # A dashboard cadastra "Caetano Resende Domingues" com o código caetano-resende-domingues.
    esperado = sha("caetano-resende-domingues")
    for escrito in ["Caetano.Resende.Domingues", "caetano resende domingues", "CAETANO_RESENDE-DOMINGUES", " Caetano  Resende Domingues "]:
        assert hash_motorista(escrito) == esperado


def test_acento_e_cedilha_saem():
    assert normalizar_id("João Conceição") == "Joao-Conceicao"
    assert hash_motorista("DRV-1013") == sha("drv-1013")


def test_dispositivo_e_tipo_de_veiculo():
    assert normalizar_dispositivo("Device_123") == "device-123"
    assert normalizar_tipo_veiculo("Carro") == "carro_app"
    assert normalizar_tipo_veiculo("Ônibus") == "onibus"
    assert normalizar_tipo_veiculo("Caminhão") == "caminhao"
    assert normalizar_tipo_veiculo("trator") == "caminhao"


def test_cliente_usa_os_ids_normalizados():
    c = DriveGuardClient("http://api", "chave", "Caetano.Resende.Domingues", device_id="device_123", tipo_veiculo="Carro")
    assert c.device_id == "device-123"
    assert c.motorista_hash == sha("caetano-resende-domingues")
    assert c.tipo_veiculo == "carro_app"
