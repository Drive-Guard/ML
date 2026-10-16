from utilitaries.DistractionDetector import DistractionDetector


def test_distracao_sustentada_gera_um_evento():
    d = DistractionDetector(min_distracao_s=2)
    eventos = []
    for q in range(100):  # 10 quadros/s, cabeça virada por 10 s
        eventos += d.atualizar(q / 10, rosto=True, yaw=40)
    assert eventos == ["distracao"]


def test_olhada_no_retrovisor_nao_conta():
    d = DistractionDetector(min_distracao_s=2)
    eventos = []
    for q in range(50):
        eventos += d.atualizar(q / 10, rosto=True, yaw=40 if 10 <= q < 18 else 0)
    assert eventos == []


def test_cabeca_baixa_com_olho_fechado_e_sono_nao_distracao():
    d = DistractionDetector(min_distracao_s=2)
    eventos = []
    for q in range(50):
        eventos += d.atualizar(q / 10, rosto=True, pitch=-30, olho_fechado=True)
    assert eventos == []


def test_rosto_fora_da_camera():
    d = DistractionDetector(min_sem_rosto_s=3)
    eventos = []
    for q in range(60):
        eventos += d.atualizar(q / 10, rosto=False)
    assert eventos == ["perda_de_face"]
