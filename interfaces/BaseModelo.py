from abc import abstractmethod,ABC
from utilitaries.Utils import Utils
from utilitaries.FeatureEngineering import FeatureEngineering
import os
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

class BaseModelo(ABC):
    def __init__(self) -> None:
        self.utils = Utils()
        self.feature_engineering = FeatureEngineering()
        self._model_name = None
        self._pipeline_step_name = None
        self.X_test = []
        self.X_train = []
        self.y_test = []
        self.y_train = []

    @property
    def pipeline_step_name(self)->str:
        return self._pipeline_step_name

    @pipeline_step_name.setter
    def pipeline_step_name(self,value:str):
        self._pipeline_step_name = value

    @property
    def model_name(self) -> str:
        return self._model_name

    @model_name.setter
    def model_name(self,value:str):
        self._model_name = value

    def train_model(self,file_path:str,model_path : str):
        if os.path.basename(file_path) == 'features_video_train.csv':
            print("Features já foram extraídas")
            df = self.utils.create_dataframe(file_path)
            self.X_train, self.X_test, self.y_train, self.y_test = self.split_data(df)
            model = self.create_model()
            pipeline = self.create_pipeline(model, self.pipeline_step_name)
            pipeline.fit(self.X_train, self.y_train)
            self.utils.save_model(pipeline, model_path)
        else:
            print("Extraindo features...")
            try:
                self.feature_engineering.extract_features_from_video(file_path)
                print('Finalizado com sucesso')
            except Exception as e:
                print('Erro ao extrair features:', e)
        pass

    def test_model(self,pipeline):
        y_pred = pipeline.predict(self.X_test)
        step_name = self.get_step_name()
        self.utils.create_model_evaluation_report(self.y_test, y_pred)
        self.utils.create_confusion_matrix(self.y_test, y_pred, pipeline.named_steps[step_name], self.model_name)

    def split_data(self,df:pd.DataFrame) -> pd.DataFrame:
        feature_columns = ["ear_behavior", "pitch_behavior", "mar_behavior", "perclos"]
        X = df[feature_columns]
        y = df["label"]
        groups = df["participant_id"]

        gfk = GroupShuffleSplit(n_splits=2, test_size=0.3, random_state=42)
        for train_idx, test_idx in gfk.split(X, y, groups):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        return X_train, X_test, y_train, y_test

    
    def create_pipeline(self,model,step_name) -> Pipeline:
        return self.utils.create_pipeline(model, step_name)

    @abstractmethod
    def create_model(self):
        pass