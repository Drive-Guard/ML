import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import FaceLandmarker
from pathlib import Path
import pandas as pd
from collections import deque
from utilitaries.Utils import Utils
from utilitaries import CloudPublisher
from utilitaries.BlinkDetector import BlinkDetector
from utilitaries.DistractionDetector import DistractionDetector
import os
import time

class FeatureEngineering:
    def __init__(self) -> None:
        self.utils = Utils()
        self.FRAME_SKIP = 3
        self.EAR_THRESHOLD = 0.2
        self.WINDOW_EAR = 10
        self.WINDOW_PITCH = 40
        self.WINDOW_MAR = 60
        self.WINDOW_PERCLOS = 1200
        self.TARGET_MINUTES = [1,2,8,9]
        self.WINDOW_SIZE = 15
        self.PROP_FRAME_WIDTH = 640
        self.PROP_FRAME_HEIGHT = 480
        self.PERCLOS_SECONDS = 60
        self.PERCLOS_THRESHOLD = 0.15
        self.YAWN_THRESHOLD = 0.60
        self.HEAD_DROP_THRESHOLD = -15
        # 3 = alerta / fadiga / sonolento (padrão); 2 = rótulo binário antigo,
        # que descartava as linhas intermediárias.
        self.NUM_CLASSES = int(os.getenv("DRIVEGUARD_CLASSES", "3"))
        self.ALERT_PERCLOS_THRESHOLD = 0.02
        self.CLOSED_MOUTH_THRESHOLD = 0.10
        self.YAW_THRESHOLD = 30
        self.ROLL_THRESHOLD = 25
        # Piscadas e distração (ver BlinkDetector e DistractionDetector).
        self.BLINK_RATE_HIGH = 30          # piscadas/min acima disso: piscadas excessivas
        self.SLOW_BLINKS_HIGH = 4          # piscadas lentas (>= 300 ms) no último minuto
        self.ALERT_COOLDOWN_S = 30         # intervalo mínimo entre alertas do mesmo tipo
        self.BLINK_COLUMNS = ['blink_rate', 'blink_duration_ms', 'slow_blinks', 'eye_closure_ms']
        base_options = python.BaseOptions(model_asset_path= 'utilitaries/model_assets/face_landmarker.task')
        self.face_landmarker_options = vision.FaceLandmarkerOptions(base_options=base_options,
                                            output_face_blendshapes=True,
                                            output_facial_transformation_matrixes=True,
                                            num_faces=1)
    
    def create_folder_map(self, folder_path: str) -> dict:
        folder = Path(folder_path)
        dataset_map = {
                0: folder / 'active',
                1: folder / 'fatigue'
        }
        return dataset_map
    
    def separate_files(self,folder_map:map) -> list:
        videos = []
        for label,folder in folder_map.items():
            files = [f for f in folder.iterdir() if f.is_file()]
            for f in files:
                videos.append(f)
        return videos

    def adjust_label_column(self,df:pd.DataFrame) -> pd.DataFrame:
        df['ear_behavior'] = df.groupby('participant_id')['ear'].transform(lambda x: x.rolling(window=10, min_periods=1).mean())
        df['pitch_behavior'] = df.groupby('participant_id')['pitch'].transform(lambda x: x.rolling(window=40, min_periods=1).median())
        df['yaw_behavior'] = df.groupby('participant_id')['yaw'].transform(lambda x: x.rolling(window=40, min_periods=1).median())
        df['roll_behavior'] = df.groupby('participant_id')['roll'].transform(lambda x: x.rolling(window=40, min_periods=1).median())
        df['mar_behavior'] = df.groupby('participant_id')['mar'].transform(lambda x: x.rolling(window=60, min_periods=1).mean())
        df['label'] = np.nan
        # Com 3 classes, o que não é alerta nem sono é fadiga (classe 1): PERCLOS
        # entre os dois limiares, boca entreaberta sem chegar ao bocejo ou cabeça
        # começando a cair. No rótulo binário essas linhas eram descartadas.
        if self.NUM_CLASSES == 3:
            df['label'] = 1

        alert_condition = (
            (df['perclos'] <= self.ALERT_PERCLOS_THRESHOLD) & 
            (df['mar_behavior'] <= self.CLOSED_MOUTH_THRESHOLD) &
            (df['pitch_behavior'] > -5.0) 
        )
        df.loc[alert_condition, 'label'] = 0

        sleep_condition = (
            (df['perclos'] >= self.PERCLOS_THRESHOLD) |
            (df['mar_behavior'] >= self.YAWN_THRESHOLD) |
            (df['pitch_behavior'] <= self.HEAD_DROP_THRESHOLD) 
        )
        df.loc[sleep_condition, 'label'] = 2 if self.NUM_CLASSES == 3 else 1
        return df.dropna(subset=['label']).copy()
    
    def fill_blink_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        As métricas de piscada só existem depois dos primeiros segundos de
        cada trecho. O treino descarta linhas com vazio (dropna), então o
        começo do trecho recebe o primeiro valor medido, e trecho sem nenhuma
        piscada fica com 0.
        """
        grupos = df.groupby(['participant_id', 'minute'])
        for coluna in self.BLINK_COLUMNS:
            df[coluna] = grupos[coluna].transform(lambda s: s.bfill().ffill()).fillna(0)
        return df

    def extract_features_from_video(self, video_path: str) -> None:
        features = []
        with FaceLandmarker.create_from_options(self.face_landmarker_options) as landmarker:
            for video in self.separate_files(self.create_folder_map(video_path)):
                participant_id = os.path.basename(video).split('.mp4')[0].split('_')[0]
                cap = cv2.VideoCapture(str(video))
                if not cap.isOpened():
                    continue
                fps = int(cap.get(cv2.CAP_PROP_FPS)) 
                if fps <= 0 or np.isnan(fps):
                    fps = 30
                frames_to_process = int(fps * 60)
                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                for target in self.TARGET_MINUTES:
                    start_frame = target * frames_to_process  
                    if start_frame >= frames:
                        break
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                    frames_read = 0
                    ear_buffer = deque(maxlen=self.WINDOW_EAR)
                    pitch_buffer = deque(maxlen=self.WINDOW_PITCH)
                    roll_buffer = deque(maxlen=self.WINDOW_PITCH)
                    yaw_buffer = deque(maxlen=self.WINDOW_PITCH)
                    mar_buffer = deque(maxlen=self.WINDOW_MAR)
                    perclos_buffer = deque(maxlen=self.WINDOW_PERCLOS)
                    blinks = BlinkDetector(limiar_ear=self.EAR_THRESHOLD)
                    while frames_read < frames_to_process:
                        ret, capture = cap.read()
                        if not ret:
                            break
                        frames_read += 1
                        if frames_read % self.FRAME_SKIP == 0:
                            continue
                        frame_rgb = cv2.cvtColor(capture, cv2.COLOR_BGR2RGB)
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                        detection_result = landmarker.detect(mp_image)

                        if detection_result.facial_transformation_matrixes and detection_result.face_landmarks:                
                            ear = self.utils.calculate_eye_aspect_ratio(detection_result.face_landmarks[0])
                            mar = self.utils.calcular_mouth_aspect_ratio(detection_result.face_landmarks[0])
                            matrix = detection_result.facial_transformation_matrixes[0]
                            pitch, yaw, roll = self.utils.extract_euler_angles(matrix)
                            ear_buffer.append(ear)
                            pitch_buffer.append(pitch)
                            yaw_buffer.append(yaw)
                            roll_buffer.append(roll)
                            mar_buffer.append(mar)
                            estado_olho = 1 if ear < self.EAR_THRESHOLD else 0
                            perclos_buffer.append(estado_olho)
                            # O EAR cru (sem média) alimenta a piscada: a média
                            # de 10 quadros apagaria uma piscada de 150 ms.
                            blinks.atualizar(ear, frames_read / fps)
                            metricas = blinks.metricas()
                            features.append({
                            'participant_id': participant_id,
                            'minute': target,
                            'frame': frames_read,
                            'ear': np.array(ear_buffer).mean(),
                            'mar': np.array(mar_buffer).mean(),
                            'pitch': np.median(pitch_buffer),
                            'yaw': np.median(yaw_buffer),
                            'roll': np.median(roll_buffer),
                            'perclos': np.array(perclos_buffer).mean(),
                            'blink_rate': metricas['blink_rate'],
                            'blink_duration_ms': metricas['duracao_media_ms'],
                            'slow_blinks': metricas['piscadas_lentas'],
                            'eye_closure_ms': metricas['duracao_olhos_fechados_ms'],
                        })
                cap.release()
        df = self.utils.create_dataframe_from_list(features, columns=["participant_id","minute","frame","ear","mar","pitch","yaw","roll","perclos"] + self.BLINK_COLUMNS)
        df = self.fill_blink_columns(df)
        df = self.adjust_label_column(df)
        self.utils.transform_dataframe_to_csv(df, 'train', 'features_video')

    def extract_features_from_webcam(self,model,cliente=None) -> None:
        loaded_model = self.utils.load_model(model)
        features_data_log = []
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.PROP_FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.PROP_FRAME_HEIGHT)
        ear_buffer = deque(maxlen=self.WINDOW_EAR)
        pitch_buffer = deque(maxlen=self.WINDOW_PITCH)
        mar_buffer = deque(maxlen=self.WINDOW_MAR)
        perclos_buffer = deque(maxlen=self.WINDOW_PERCLOS)
        roll_buffer = deque(maxlen=self.WINDOW_PITCH)
        yaw_buffer = deque(maxlen=self.WINDOW_PITCH)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        ultimo_envio_nuvem = 0.0
        estado_anterior = 'alerta'
        blinks = BlinkDetector(limiar_ear=self.EAR_THRESHOLD)
        distracao = DistractionDetector(limiar_yaw=self.YAW_THRESHOLD)
        ultimo_alerta = {}
        aviso_tela = ('', 0.0)

        with FaceLandmarker.create_from_options(self.face_landmarker_options) as landmarker:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("Não foi possível acessar a webcam.")
                    break
                    
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face = face_cascade.detectMultiScale(frame_rgb,scaleFactor=1.1,minNeighbors=5)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                detection = landmarker.detect(mp_image)
                
                state = 'Acordado'
                message_color = (0, 255, 0) 
                
                agora = time.monotonic()
                eventos = []
                if not (detection and detection.face_landmarks):
                    eventos += distracao.atualizar(agora, rosto=False)
                if detection and detection.face_landmarks:
                    ear = self.utils.calculate_eye_aspect_ratio(detection.face_landmarks[0])
                    mar = self.utils.calcular_mouth_aspect_ratio(detection.face_landmarks[0])
                    matrix = detection.facial_transformation_matrixes[0]
                    pitch, yaw, roll = self.utils.extract_euler_angles(matrix)
                    ear_buffer.append(ear)
                    pitch_buffer.append(pitch)
                    yaw_buffer.append(yaw)
                    roll_buffer.append(roll)
                    mar_buffer.append(mar)
                    perclos_buffer.append(1 if ear < self.EAR_THRESHOLD else 0)
                    blinks.atualizar(ear, agora)
                    if blinks.novo_fechamento_longo:
                        eventos.append('fechamento_prolongado')
                    eventos += distracao.atualizar(agora, rosto=True, yaw=float(yaw), pitch=float(pitch), olho_fechado=ear < self.EAR_THRESHOLD)
                    
                    if len(ear_buffer) >= self.WINDOW_EAR:                    
                        ear_comp = np.array(ear_buffer).mean()
                        mar_comp = np.array(mar_buffer).mean()
                        pitch_comp = np.median(pitch_buffer)
                        yaw_comp = np.median(yaw_buffer)
                        roll_comp = np.median(roll_buffer)
                        perclos_val = np.array(perclos_buffer).mean()
                        piscadas = blinks.metricas()
                        if (piscadas['blink_rate'] or 0) >= self.BLINK_RATE_HIGH or piscadas['piscadas_lentas'] >= self.SLOW_BLINKS_HIGH:
                            eventos.append('piscadas_excessivas')
                        
                        X_teste = pd.DataFrame([{
                            'ear_behavior': ear_comp,
                            'pitch_behavior': pitch_comp,
                            'roll_behavior': roll_comp,
                        }])
                        X = X_teste[['ear_behavior','pitch_behavior','roll_behavior']]
                        estado, score = CloudPublisher.avaliar(loaded_model, X)[0]
                        predicao = CloudPublisher.CODIGO_ESTADO[estado]
                        features_data_log.append([ear_comp,pitch_comp,roll_comp,predicao])
                        if cliente is not None:
                            if time.monotonic() - ultimo_envio_nuvem >= 1.0:
                                ultimo_envio_nuvem = time.monotonic()
                                self._enviar_para_nuvem(cliente, estado, score, estado_anterior,
                                                        ear_comp, mar_comp, perclos_val, pitch_comp, yaw_comp, roll_comp,
                                                        piscadas['blink_rate'], piscadas['duracao_olhos_fechados_ms'])
                                estado_anterior = estado
                        if estado == 'sonolento':
                            state = "SONO!"
                            message_color = (0, 0, 255)
                        elif estado == 'fadiga':
                            state = "FADIGA"
                            message_color = (0, 165, 255)
                        
                        cv2.putText(frame, f"EAR: {ear_comp:.2f}", (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, message_color, 2)
                        cv2.putText(frame, f"MAR: {mar_comp:.2f}", (10, 45), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, message_color, 2)
                        cv2.putText(frame, f"PITCH: {pitch_comp:.2f}", (10, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, message_color, 2)
                        cv2.putText(frame, f"YAW: {yaw_comp:.2f}", (10, 75), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, message_color, 2)
                        cv2.putText(frame, f"ROLL: {roll_comp:.2f}", (10, 90), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, message_color, 2)
                        cv2.putText(frame, f"PERCLOS: {perclos_val:.2f}", (10, 105), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, message_color, 2)
                        taxa = piscadas['blink_rate']
                        cv2.putText(frame, f"PISCADAS/MIN: {taxa:.0f}" if taxa is not None else "PISCADAS/MIN: ...", (10, 120), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, message_color, 2)
                        cv2.putText(frame, f"OLHO FECHADO: {blinks.olho_fechado_ms:.0f} ms", (10, 135), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, message_color, 2)
                        
                # Alertas próprios da piscada e da distração: independem do
                # classificador e saem com intervalo mínimo por tipo.
                for evento in dict.fromkeys(eventos):
                    if agora - ultimo_alerta.get(evento, -1e9) < self.ALERT_COOLDOWN_S:
                        continue
                    ultimo_alerta[evento] = agora
                    aviso_tela = (self.MENSAGENS_EVENTO[evento], agora)
                    if cliente is not None:
                        self._registrar_evento(cliente, evento, blinks.metricas())
                if aviso_tela[0] and agora - aviso_tela[1] < 3:
                    cv2.putText(frame, aviso_tela[0], (10, 200), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 165, 255), 2)

                cv2.putText(frame, f"{state}", (10, 165), cv2.FONT_HERSHEY_COMPLEX, 1, message_color, 2)
                for x,y,w,h in face:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), message_color, 2)
                
                cv2.imshow('Monitoramento de sono', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()
        if features_data_log:
            df_log = self.utils.create_dataframe_from_list(features_data_log,['ear_behavior','pitch_behavior','roll_behavior','prediction'])
            predictions = df_log.pop("prediction").to_numpy()
            self.utils.generate_model_test_log(df_log, predictions, 'label')

    MENSAGENS_EVENTO = {
        'fechamento_prolongado': 'OLHOS FECHADOS!',
        'piscadas_excessivas': 'PISCADAS LENTAS/FREQUENTES',
        'distracao': 'OLHE PARA A ESTRADA',
        'perda_de_face': 'ROSTO FORA DA CAMERA',
    }

    GRAVIDADE_EVENTO = {
        'fechamento_prolongado': ('critica', 92.0),
        'piscadas_excessivas': ('media', 65.0),
        'distracao': ('alta', 75.0),
        'perda_de_face': ('media', 60.0),
    }

    def _registrar_evento(self, cliente, evento: str, piscadas: dict) -> None:
        """Registra como alerta um evento de piscada ou distração."""
        try:
            gravidade, score = self.GRAVIDADE_EVENTO[evento]
            cliente.registrar_alerta(
                tipo=evento,
                gravidade=gravidade,
                score_fadiga=score,
                mensagem=(
                    f"{piscadas['blink_rate'] or 0:.0f} piscadas/min, {piscadas['piscadas_lentas']} lentas, "
                    f"fechamento de {piscadas['duracao_olhos_fechados_ms']} ms."
                ),
            )
        except Exception as e:
            print(f"Aviso: falha ao registrar alerta para a AWS: {e}")

    def _enviar_para_nuvem(self, cliente, estado, score, estado_anterior,
                           ear, mar, perclos, pitch, yaw, roll, blink_rate=None, duracao_olhos_fechados_ms=None) -> None:
        """
        Registra a leitura no cliente da AWS (uma por segundo, chamado pelo loop).

        O envio em si roda em outra thread dentro do cliente: a webcam não
        espera a rede. Qualquer falha aqui é registrada e ignorada, para não
        derrubar o monitoramento.
        """
        try:
            cliente.registrar_leitura(
                ear=ear, mar=mar, perclos=perclos,
                head_pitch=pitch, head_yaw=yaw, head_roll=roll,
                score_fadiga=score, estado=estado,
                blink_rate=blink_rate, duracao_olhos_fechados_ms=duracao_olhos_fechados_ms,
            )
            if estado == 'sonolento' and estado_anterior != 'sonolento':
                cliente.registrar_alerta(
                    tipo=CloudPublisher.tipo_alerta(perclos, mar, pitch, self),
                    gravidade=CloudPublisher.gravidade(score),
                    score_fadiga=score,
                    mensagem=f"PERCLOS {perclos:.2f}, MAR {mar:.2f}, pitch {pitch:.1f} graus.",
                )
        except Exception as e:
            print(f"Aviso: falha ao registrar leitura para a AWS: {e}")
