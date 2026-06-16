import os
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

class Utils:
    def __init__(self) -> None:
        pass
    def create_dataframe(self,file_path : str) -> pd.DataFrame:
        try:
            df = pd.read_csv(file_path,delimiter=';', encoding='utf-8', low_memory=False)
            df.convert_dtypes()
            df.dropna(inplace=True)
            df.drop_duplicates(inplace=True)
            return pd.DataFrame(df)
        except Exception as e:
            print(f"Erro ao criar DataFrame: {e}")
    
    def create_dataframe_from_list(self, data: list, columns: list) -> pd.DataFrame:
        try:
            df = pd.DataFrame(data, columns=columns)
            return df
        except Exception as e:
            print(f"Erro ao criar DataFrame a partir da lista: {e}")
        
    def transform_dataframe_to_csv(self, df : pd.DataFrame, stage: str, filename : str) -> None:
        if not os.path.exists(f"data/{filename}_{stage}.csv"):
            os.makedirs(f"data/{filename}_{stage}.csv")
        file_name = f"data/{filename}_{stage}.csv"
        df.to_csv(file_name, index=False,sep=';', encoding='utf-8')
        return file_name
    
    def drop_columns(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        return df.drop(columns=columns, inplace=True)
    
    def order_by_column(self, df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        return df.sort_values(by=column_name)
    
    def create_train_test_split(self, df: pd.DataFrame, columns: list, target_column: str, test_size: float = 0.3, random_state: int = 42):
        X = df[columns]
        y = df[target_column]
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    def create_model_evaluation_report(self, y_true, y_pred,model_name:str) -> None:
        print(f"\nRelatório do modelo : {model_name}")
        print("=" * 67)
        print(classification_report(y_true, y_pred))
        print("=" * 67)

    def create_confusion_matrix(self, y_true, y_pred,model,model_name:str) -> None:
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        disp.plot(cmap=plt.cm.Greens)
        plt.title(f'Matriz de Confusão para {model_name} (Acurácia {accuracy_score(y_true, y_pred):.2%})')
        plt.show()

    def calculate_euclidian_distance(self,p1, p2) -> float:
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def calculate_eye_aspect_ratio(self,landmark_list):
        left_sup = (landmark_list[159].x, landmark_list[159].y)
        left_inf = (landmark_list[145].x, landmark_list[145].y)
        left_esq = (landmark_list[33].x, landmark_list[33].y)
        left_dir = (landmark_list[133].x, landmark_list[133].y)

        right_sup = (landmark_list[386].x, landmark_list[386].y)
        right_inf = (landmark_list[374].x, landmark_list[374].y)
        right_esq = (landmark_list[362].x, landmark_list[362].y)
        right_dir = (landmark_list[263].x, landmark_list[263].y)
        
        ear_left = self.calculate_euclidian_distance(left_sup, left_inf) / self.calculate_euclidian_distance(left_esq, left_dir)
        ear_right = self.calculate_euclidian_distance(right_sup, right_inf) / self.calculate_euclidian_distance(right_esq, right_dir)

        return (ear_left + ear_right) / 2.0

    def calcular_mouth_aspect_ratio(self,landmark_list):
        mouth_sup = (landmark_list[13].x, landmark_list[13].y)
        mouth_inf = (landmark_list[14].x, landmark_list[14].y)
        mouth_left = (landmark_list[78].x, landmark_list[78].y)
        mouth_right = (landmark_list[308].x, landmark_list[308].y)
        
        mar = self.calculate_euclidian_distance(mouth_sup, mouth_inf) / self.calculate_euclidian_distance(mouth_left, mouth_right)
        return mar
    def extract_euler_angles(matrix_4x4):
        R = matrix_4x4[0:3, 0:3]
        proj_matrix = np.hstack((R, np.zeros((3, 1))))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)
        
        pitch, yaw, roll = euler_angles.flatten()
        return pitch, yaw, roll
