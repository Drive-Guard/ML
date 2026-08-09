import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import FaceLandmarker, drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
from pathlib import Path
import pandas as pd
import hashlib
from collections import deque
from utilitaries.Utils import Utils

class FeatureEngineering:
    def __init__(self) -> None:
        self.utils = Utils()
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
    
    def extract_features_from_image(self,filepath:str) -> None:
        options = self.face_landmarker_options
        dataset_map = self.create_folder_map(filepath)

        features_data = []
        hash_img = set()
        ignored_images = 0

        df = pd.DataFrame(columns=["ear", "mar", "pitch", "yaw", "roll", "label"])
        with FaceLandmarker.create_from_options(options) as landmarker:
            for label, folder in dataset_map.items():
                if folder is None or not folder.exists() or not folder.is_dir():
                    print(f"Aviso: Diretório {folder} não encontrado. Pulando para o próximo")
                    continue
                
                files = sorted([f for f in folder.iterdir() if f.is_file()])    
                for file in files:
                    with open(file, "rb") as f:
                        file_hash = hashlib.md5(f.read()).hexdigest()
                    
                    if file_hash in hash_img:
                        ignored_images += 1
                        continue

                    hash_img.add(file_hash)
                    img = cv2.imread(file)
                    if img is not None:
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                        detection_result = landmarker.detect(mp_image)

                        if detection_result.facial_transformation_matrixes and detection_result.face_landmarks:
                            matrix = detection_result.facial_transformation_matrixes[0]
                            pitch, yaw, roll = self.utils.extract_euler_angles(matrix)
                            ear = self.utils.calculate_eye_aspect_ratio(detection_result.face_landmarks[0])
                            mar = self.utils.calcular_mouth_aspect_ratio(detection_result.face_landmarks[0])

                            features_data.append({
                                "ear": ear,
                                "mar": mar,
                                "pitch": pitch,
                                "yaw": yaw,
                                "roll": roll,
                                "label": label
                            })

        df = self.utils.create_dataframe_from_list(features_data, columns=["ear", "mar", "pitch", "yaw", "roll", "label"])
        self.utils.transform_dataframe_to_csv(df, 'trusted', 'features_data')

    def extract_features_from_video(self, video_path: str) -> pd.DataFrame:
        options = self.face_landmarker_options
        dataset_map = self.create_folder_map(video_path)

        features_data = []
        hash_img = set()
        ignored_images = 0

        df = pd.DataFrame(columns=["ear", "mar", "pitch", "yaw", "roll", "label"])
        with FaceLandmarker.create_from_options(options) as landmarker:
            for label, folder in dataset_map.items():
                if folder is None or not folder.exists() or not folder.is_dir():
                    print(f"Aviso: Diretório {folder} não encontrado. Pulando para o próximo")
                    continue
                
                files = sorted([f for f in folder.iterdir() if f.is_file()])    
                for file in files:
                    with open(file, "rb") as f:
                        file_hash = hashlib.md5(f.read()).hexdigest()
                    
                    if file_hash in hash_img:
                        ignored_images += 1
                        continue

                    hash_img.add(file_hash)
                    img = cv2.imread(file)
                    if img is not None:
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                        detection_result = landmarker.detect(mp_image)

                        if detection_result.facial_transformation_matrixes and detection_result.face_blendshapes:
                            matrix = detection_result.facial_transformation_matrixes[0]
                            pitch, yaw, roll = self.utils.extract_euler_angles(matrix)
                            ear = self.utils.calculate_eye_aspect_ratio(detection_result.face_landmarks[0])
                            mar = self.utils.calcular_mouth_aspect_ratio(detection_result.face_landmarks[0])

                            features_data.append({
                                "ear": ear,
                                "mar": mar,
                                "pitch": pitch,
                                "yaw": yaw,
                                "roll": roll,
                                "label": label
                            })

        df = self.utils.create_dataframe_from_list(features_data, columns=["ear", "mar", "pitch", "yaw", "roll", "label"])
        self.utils.transform_dataframe_to_csv(df, 'trusted', 'features_data')

    def extract_features_from_webcam(self) -> None:
        options = self.face_landmarker_options
        features_data = []
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        USER_ID = 100
        WINDOW_SIZE = 15

        with FaceLandmarker.create_from_options(options) as landmarker:
            if not cap.isOpened():
                print("Erro ao abrir a webcam")
                exit()

            window_ear = deque(maxlen=WINDOW_SIZE)
            window_mar = deque(maxlen=WINDOW_SIZE)  
            window_pitch = deque(maxlen=WINDOW_SIZE)
            window_yaw = deque(maxlen=WINDOW_SIZE)
            window_roll = deque(maxlen=WINDOW_SIZE)

            historico_olhos = []

        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)

            if not ret:
                print("Error: Failed to grab a frame.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            detection_result = landmarker.detect(mp_image)

            if detection_result.facial_transformation_matrixes and detection_result.face_landmarks:                
                ear = self.utils.calculate_eye_aspect_ratio(detection_result.face_landmarks[0])
                mar = self.utils.calcular_mouth_aspect_ratio(detection_result.face_landmarks[0])
                matrix = detection_result.facial_transformation_matrixes[0]
                pitch, yaw, roll = self.utils.extract_euler_angles(matrix)
                historico_olhos.append(ear)

                window_ear.append(ear)
                window_mar.append(mar)
                window_pitch.append(pitch)
                window_roll.append(roll)
                window_yaw.append(yaw)

                if len(window_ear) == WINDOW_SIZE:
                    arr_ear = np.array(window_ear,dtype=float)
                    arr_mar = np.array(window_mar,dtype=float)
                    arr_pitch = np.array(window_pitch,dtype=float)
                    arr_roll = np.array(window_roll,dtype=float)
                    arr_yaw = np.array(window_yaw,dtype=float)

                    features_data.append({
                        "participant_id": USER_ID,
                        "ear": arr_ear.mean(),
                        "pitch": arr_pitch.mean(),
                        "perclos": sum(historico_olhos) / WINDOW_SIZE,
                        "mar": arr_mar.mean(),
                        "yaw": arr_yaw.mean(),
                        "roll": arr_roll.mean(),
                    })  

            cv2.imshow('Webcam', cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(1) 
            if key == ord('q'):
                break
        cap.release()
        cap.destroyAllWindows()