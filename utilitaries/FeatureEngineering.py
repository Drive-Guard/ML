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
import os

class FeatureEngineering:
    def __init__(self) -> None:
        self.utils = Utils()
        self.FRAME_SKIP = 3
        self.EAR_THRESHOLD = 0.2
        self.WINDOW_EAR = 10
        self.WINDOW_PITCH = 40
        self.WINDOW_MAR = 60
        self.WINDOW_PERCLOS = 1200
        self.TARGET_MINUTES = [1,2,3]
        self.WINDOW_SIZE = 15
        self.PROP_FRAME_WIDTH = 640
        self.PROP_FRAME_HEIGHT = 480
        self.PERCLOS_SECONDS = 60
        self.PERCLOS_THRESHOLD = 0.15
        self.YAWN_THRESHOLD = 0.6
        self.HEAD_DROP_THRESHOLD = 20
        self.ALERT_PERCLOS_THRESHOLD = 0.02
        self.CLOSED_MOUTH_THRESHOLD = 0.1
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
        for label, folder in folder_map.items():
            files = [f for f in folder.iterdir() if f.is_file()]
            for f in files:
                videos.append(f)
        return videos

    def adjust_label_column(self,df:pd.DataFrame) -> pd.DataFrame:
        df['ear_behavior'] = df.groupby('participant_id')['ear'].transform(lambda x: x.rolling(window=10, min_periods=1).mean())
        df['pitch_behavior'] = df.groupby('participant_id')['pitch'].transform(lambda x: x.rolling(window=40, min_periods=1).median())
        df['mar_behavior'] = df.groupby('participant_id')['mar'].transform(lambda x: x.rolling(window=60, min_periods=1).mean())
        df['label'] = np.nan

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
        df.loc[sleep_condition, 'label'] = 1
        return df.dropna(subset=['label']).copy()
    
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
                            roll_buffer.append(roll)
                            mar_buffer.append(mar)
                            estado_olho = 1 if ear < self.EAR_THRESHOLD else 0
                            perclos_buffer.append(estado_olho)
                            features.append({
                            'participant_id': participant_id,
                            'minute': target,
                            'frame': frames_read,
                            'ear': np.array(ear_buffer).mean(),
                            'mar': np.array(mar_buffer).mean(),
                            'pitch': np.median(pitch_buffer),
                            'roll': np.array(roll_buffer).mean(),
                            'perclos': np.array(perclos_buffer).mean()
                        })
                cap.release()
        df = self.utils.create_dataframe_from_list(features, columns=["participant_id","minute","frame","ear","mar","pitch","roll","perclos"])
        df = self.adjust_label_column(df)
        self.utils.transform_dataframe_to_csv(df, 'train', 'features_video')

    def extract_features_from_webcam(self,model) -> None:
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
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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
                
                if detection and detection.face_landmarks:
                    ear = self.utils.calculate_eye_aspect_ratio(detection.face_landmarks[0])
                    mar = self.utils.calcular_mouth_aspect_ratio(detection.face_landmarks[0])
                    matrix = detection.facial_transformation_matrixes[0]
                    pitch, yaw, roll = self.utils.extract_euler_angles(matrix)
                    ear_buffer.append(ear)
                    pitch_buffer.append(pitch)
                    roll_buffer.append(roll)
                    mar_buffer.append(mar)
                    perclos_buffer.append(1 if ear < self.EAR_THRESHOLD else 0)
                    
                    if len(ear_buffer) >= self.WINDOW_EAR:                    
                        ear_comp = np.array(ear_buffer).mean()
                        mar_comp = np.array(mar_buffer).mean()
                        pitch_comp = np.median(pitch_buffer)
                        roll_comp = np.array(roll_buffer).mean()
                        perclos_val = np.array(perclos_buffer).mean()
                        
                        X_teste = pd.DataFrame([{
                            'ear_behavior': ear_comp,
                            'pitch_behavior': pitch_comp,
                            'mar_behavior': mar_comp,
                            'perclos': perclos_val
                        }])
                        X = X_teste[['ear_behavior','pitch_behavior','mar_behavior','perclos']]
                        pred = int(loaded_model.predict(X)[0])
                        predicao = 1 if pred == 1 else 0
                        
                        if predicao == 1:
                            state = "SONO!"
                            message_color = (0, 0, 255)
                        
                        cv2.putText(frame, f"EAR: {ear_comp:.2f}", (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, message_color, 2)
                        cv2.putText(frame, f"MAR: {mar_comp:.2f}", (10, 45), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, message_color, 2)
                        cv2.putText(frame, f"PITCH: {pitch_comp:.2f}", (10, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, message_color, 2)
                        cv2.putText(frame, f"ROLL: {roll_comp:.2f}", (10, 75), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, message_color, 2)
                        cv2.putText(frame, f"PERCLOS: {perclos_val:.2f}", (10, 90), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, message_color, 2)
                        
                cv2.putText(frame, f"{state}", (10, 165), cv2.FONT_HERSHEY_COMPLEX, 1, message_color, 2)
                for x,y,w,h in face:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), message_color, 2)
                
                cv2.imshow('Monitoramento de sono', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

        if features_data_log:
            df_log = self.utils.create_dataframe_from_list(features_data_log,['ear_behavior','pitch_behavior','mar_behavior','perclos','prediction'])
            predictions = df_log.pop("prediction").to_numpy()
            self.utils.generate_model_test_log(df_log, predictions, 'label')