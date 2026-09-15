"""
Pipeline de inferência em tempo real — o que roda dentro do veículo.

    Câmera/vídeo
      → MediaPipe Face Mesh (468 landmarks)
      → EAR, MAR, head pose + PERCLOS e blink rate em janela deslizante
      → VotingClassifier (ensemble treinado em DrowsinessDetection)
      → score contínuo 0-100 + alerta local progressivo
      → DriveGuardClient → API Gateway → S3 Bronze → RDS → dashboard

Duas garantias de projeto:

* **O alerta não depende da nuvem.** Ele é emitido assim que o score cruza o
  limiar, no próprio laço. O envio para a AWS acontece depois e em lote; se a
  rede cair, a telemetria fica no buffer em disco e o motorista continua
  protegido.

* **Nenhuma imagem sai do veículo.** O frame é processado e descartado na
  mesma iteração. Só os números anonimizados sobem.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from datetime import datetime, timezone

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import FaceLandmarker

from models.DrowsinessDetection import DrowsinessDetection
from utilitaries.DriveGuardClient import DriveGuardClient
from utilitaries.Utils import Utils

logger = logging.getLogger(__name__)


class DrowsinessStream:
    """Executa o modelo de votação sobre um fluxo de vídeo ao vivo."""

    def __init__(
        self,
        fonte: int | str = 0,
        janela_segundos: int = 60,
        fps_estimado: int = 30,
        intervalo_envio_segundos: int = 15,
        limiar_alerta: float = 60.0,
        limiar_critico: float = 80.0,
        ear_fechado: float = 0.20,
        exibir_video: bool = True,
        modelo: DrowsinessDetection | None = None,
        cliente: DriveGuardClient | None = None,
    ) -> None:
        self.fonte = fonte
        self.janela_segundos = janela_segundos
        self.fps_estimado = fps_estimado
        self.intervalo_envio = intervalo_envio_segundos
        self.limiar_alerta = limiar_alerta
        self.limiar_critico = limiar_critico
        self.ear_fechado = ear_fechado
        self.exibir_video = exibir_video

        self.utils = Utils()
        self.modelo = modelo or DrowsinessDetection()
        self.cliente = cliente or DriveGuardClient()

        # Janela deslizante de EAR: base do PERCLOS e da taxa de piscadas.
        tamanho = janela_segundos * fps_estimado
        self._ear_janela: deque[float] = deque(maxlen=tamanho)
        self._olho_fechado_anterior = False
        self._piscadas: deque[float] = deque(maxlen=200)
        self._inicio_fechamento: float | None = None
        self._duracoes_fechamento: deque[float] = deque(maxlen=50)

        # Baseline adaptativo: cada motorista tem uma abertura ocular própria,
        # então o limiar de "olho fechado" é calibrado nos primeiros segundos
        # em vez de usar um valor fixo da literatura.
        self._baseline_ear: float | None = None
        self._amostras_baseline: list[float] = []
        self.segundos_calibracao = 10

        base_options = python.BaseOptions(
            model_asset_path="utilitaries/model_assets/face_landmarker.task"
        )
        self._opcoes = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
        )

    # ------------------------------------------------------- metricas temporais

    def _atualizar_janela(self, ear: float) -> None:
        agora = time.monotonic()
        self._ear_janela.append(ear)

        if self._baseline_ear is None:
            self._amostras_baseline.append(ear)
            if len(self._amostras_baseline) >= self.segundos_calibracao * self.fps_estimado:
                # Mediana em vez de média: ignora os frames em que o motorista
                # piscou durante a calibração.
                amostras = sorted(self._amostras_baseline)
                self._baseline_ear = amostras[len(amostras) // 2]
                logger.info("Baseline de EAR calibrado em %.4f", self._baseline_ear)

        fechado = ear < self._limiar_fechado()

        if fechado and not self._olho_fechado_anterior:
            self._inicio_fechamento = agora
        elif not fechado and self._olho_fechado_anterior:
            self._piscadas.append(agora)
            if self._inicio_fechamento is not None:
                self._duracoes_fechamento.append(agora - self._inicio_fechamento)
            self._inicio_fechamento = None

        self._olho_fechado_anterior = fechado

    def _limiar_fechado(self) -> float:
        # 70% do baseline do motorista; antes da calibração, o valor da literatura.
        if self._baseline_ear is None:
            return self.ear_fechado
        return self._baseline_ear * 0.70

    def _perclos(self) -> float:
        """Percentual de tempo com os olhos fechados na janela."""
        if not self._ear_janela:
            return 0.0
        limiar = self._limiar_fechado()
        fechados = sum(1 for e in self._ear_janela if e < limiar)
        return fechados / len(self._ear_janela)

    def _blink_rate(self) -> float:
        """Piscadas por minuto, contadas na janela."""
        agora = time.monotonic()
        recentes = [t for t in self._piscadas if agora - t <= self.janela_segundos]
        if not recentes:
            return 0.0
        return len(recentes) * (60.0 / self.janela_segundos)

    def _duracao_media_fechamento_ms(self) -> int:
        if not self._duracoes_fechamento:
            return 0
        return int(sum(self._duracoes_fechamento) / len(self._duracoes_fechamento) * 1000)

    # ------------------------------------------------------------------ alerta

    def _emitir_alerta_local(self, score: float, estado: str) -> None:
        """
        Alerta no veículo. Roda antes de qualquer envio à nuvem.

        Aqui fica só o aviso em console; o dispositivo real aciona o buzzer e o
        LED pelo GPIO. É este ponto que precisa funcionar offline.
        """
        gravidade = self.modelo.classificar_gravidade(score)
        marcador = "!!!" if score >= self.limiar_critico else "!"
        logger.warning(
            "%s ALERTA %s - estado=%s score=%.1f", marcador, gravidade.upper(), estado, score
        )
        print(f"\a{marcador} FADIGA DETECTADA: {estado} (score {score:.1f})")

    # -------------------------------------------------------------------- loop

    def executar(self, duracao_maxima_segundos: int | None = None) -> dict:
        self.modelo.carregar_modelo()

        captura = cv2.VideoCapture(self.fonte)
        if not captura.isOpened():
            raise RuntimeError(f"Nao foi possivel abrir a fonte de video: {self.fonte}")

        inicio = time.monotonic()
        ultimo_envio = inicio
        ultimo_alerta = 0.0
        frames, leituras = 0, 0

        logger.info(
            "Iniciando monitoramento (fonte=%s, envio a cada %ds).",
            self.fonte, self.intervalo_envio,
        )

        try:
            with FaceLandmarker.create_from_options(self._opcoes) as landmarker:
                while True:
                    ok, frame = captura.read()
                    if not ok:
                        break

                    frames += 1
                    agora = time.monotonic()

                    if duracao_maxima_segundos and agora - inicio > duracao_maxima_segundos:
                        break

                    imagem = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    )
                    resultado = landmarker.detect(imagem)

                    if not resultado.face_landmarks:
                        if self.exibir_video:
                            cv2.imshow("DriveGuard", frame)
                            if cv2.waitKey(1) & 0xFF == ord("q"):
                                break
                        continue

                    landmarks = resultado.face_landmarks[0]
                    ear = self.utils.calculate_eye_aspect_ratio(landmarks)
                    mar = self.utils.calcular_mouth_aspect_ratio(landmarks)

                    pitch = yaw = roll = 0.0
                    if resultado.facial_transformation_matrixes:
                        pitch, yaw, roll = self.utils.extract_euler_angles(
                            resultado.facial_transformation_matrixes[0]
                        )

                    self._atualizar_janela(ear)

                    predicao = self.modelo.prever(
                        {"ear": ear, "mar": mar, "pitch": pitch, "yaw": yaw, "roll": roll}
                    )
                    score = predicao["score_fadiga"]
                    estado = predicao["estado"]

                    # Uma leitura por segundo: o dashboard não precisa de 30/s,
                    # e isso divide por 30 o volume que trafega e é armazenado.
                    if frames % self.fps_estimado == 0:
                        leituras += 1
                        self.cliente.registrar_leitura(
                            ear=ear,
                            mar=mar,
                            perclos=self._perclos(),
                            blink_rate=self._blink_rate(),
                            duracao_olhos_fechados_ms=self._duracao_media_fechamento_ms(),
                            head_pitch=pitch,
                            head_yaw=yaw,
                            head_roll=roll,
                            score_fadiga=score,
                            estado=estado,
                            registrado_em=datetime.now(timezone.utc),
                        )

                    # Alerta local, no máximo um a cada 15s para não virar ruído.
                    if score >= self.limiar_alerta and agora - ultimo_alerta > 15:
                        ultimo_alerta = agora
                        self._emitir_alerta_local(score, estado)
                        self.cliente.registrar_alerta(
                            tipo=self._tipo_alerta(self._perclos(), mar, pitch),
                            gravidade=self.modelo.classificar_gravidade(score),
                            score_fadiga=score,
                            mensagem=(
                                f"PERCLOS {self._perclos():.2f}, MAR {mar:.2f}, "
                                f"pitch {pitch:.1f} graus."
                            ),
                        )

                    if agora - ultimo_envio >= self.intervalo_envio:
                        ultimo_envio = agora
                        self.cliente.enviar()

                    if self.exibir_video:
                        self._desenhar_hud(frame, score, estado, ear, mar)
                        cv2.imshow("DriveGuard", frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
        finally:
            captura.release()
            if self.exibir_video:
                cv2.destroyAllWindows()
            self.cliente.enviar(forcar=True)
            self.cliente.drenar_buffer_offline()

        resumo = {
            "frames": frames,
            "leituras_geradas": leituras,
            "leituras_enviadas": self.cliente.enviados,
            "leituras_em_buffer": self.cliente.falhas,
            "duracao_s": round(time.monotonic() - inicio, 1),
        }
        logger.info("Monitoramento encerrado: %s", resumo)
        return resumo

    @staticmethod
    def _tipo_alerta(perclos: float, mar: float, pitch: float) -> str:
        """Escolhe o tipo do alerta pela feature dominante, para o dashboard."""
        if perclos > 0.50:
            return "microssono"
        if mar > 0.50:
            return "bocejo_excessivo"
        if pitch < -15:
            return "cabeca_baixa"
        return "sonolencia"

    def _desenhar_hud(self, frame, score: float, estado: str, ear: float, mar: float) -> None:
        cor = (0, 200, 0) if score < self.limiar_alerta else (
            (0, 165, 255) if score < self.limiar_critico else (0, 0, 255)
        )
        linhas = [
            f"Estado: {estado}",
            f"Score:  {score:5.1f}",
            f"EAR:    {ear:.3f}   MAR: {mar:.3f}",
            f"PERCLOS:{self._perclos():.2f}  Piscadas/min: {self._blink_rate():.0f}",
        ]
        for i, texto in enumerate(linhas):
            cv2.putText(
                frame, texto, (12, 30 + i * 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2, cv2.LINE_AA,
            )
