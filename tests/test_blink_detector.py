"""BlinkDetector com séries de EAR sintéticas a 30 quadros por segundo."""

from utilitaries.BlinkDetector import BlinkDetector

FPS = 30


def alimentar(det, segundos, fechamentos):
    """fechamentos: lista de (inicio_s, duracao_ms) com o olho fechado."""
    for q in range(int(segundos * FPS)):
        t = q / FPS
        fechado = any(i <= t < i + d / 1000 for i, d in fechamentos)
        det.atualizar(0.12 if fechado else 0.30, t)


def test_conta_piscadas_e_taxa_por_minuto():
    det = BlinkDetector()
    alimentar(det, 60, [(i * 3.0 + 1, 150) for i in range(20)])  # 20 piscadas de 150 ms
    m = det.metricas()
    assert 19 <= m["blink_rate"] <= 21
    assert 130 <= m["duracao_media_ms"] <= 200
    assert m["piscadas_lentas"] == 0
    assert m["fechamentos_longos"] == 0


def test_piscada_lenta_e_fechamento_longo():
    det = BlinkDetector()
    alimentar(det, 30, [(2, 400), (6, 450), (10, 1500)])
    m = det.metricas()
    assert m["piscadas_lentas"] == 2
    assert m["fechamentos_longos"] == 1
    assert m["duracao_olhos_fechados_ms"] >= 1450


def test_fechamento_longo_dispara_durante_o_fechamento():
    det = BlinkDetector()
    disparos = []
    for q in range(3 * FPS):
        t = q / FPS
        det.atualizar(0.10 if t >= 1 else 0.30, t)
        if det.novo_fechamento_longo:
            disparos.append(t)
    assert len(disparos) == 1 and 2.0 <= disparos[0] < 2.1  # 1 s depois de fechar, olho ainda fechado


def test_ruido_de_um_quadro_nao_e_piscada():
    det = BlinkDetector()
    alimentar(det, 20, [(5, 20)])  # 20 ms: menos que um quadro
    assert det.metricas()["blink_rate"] == 0


def test_janela_descarta_piscadas_antigas():
    det = BlinkDetector(janela_s=60)
    alimentar(det, 120, [(i * 2.0, 150) for i in range(30)])  # só nos primeiros 60 s
    assert det.metricas()["blink_rate"] == 0
