from interfaces.BaseModelo import BaseModelo
import os
import joblib
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
class DrowsinessDetection(BaseModelo):
    def __init__(self) -> None:
        super().__init__()
        self.X_test = []
        self.X_train = []
        self.y_test = []
        self.y_train = []

    def train_model(self,file_path:str,model_path:str):
        if os.path.basename(file_path) == 'features_data_video_train.csv':
            print("Features já foram extraídas")

            df = self.utils.create_dataframe(file_path)
            self.X_train, self.X_test, self.y_train, self.y_test = self.split_data(df)
            pipeline = self.create_pipeline()
            pipeline.fit(self.X_train, self.y_train)
            self.test_model(pipeline)
            self.save_model(pipeline,model_path)
        else:
            print("Extraindo features...")
            try:
                feature_engineering = self.feature_engineering
                feature_engineering.extract_features_from_video(file_path)
                print('Finalizado com sucesso')
            except Exception as e:
                print('Erro ao extrair features:', e)
    
    def test_model(self,pipeline):
        print('Testando modelo')
        y_pred = pipeline.predict(self.X_test)
        self.utils.create_model_evaluation_report(self.y_test, y_pred , 'Random Forest')
        self.utils.create_confusion_matrix(self.y_test, y_pred, pipeline.named_steps['rf'],'Random Forest')

    def split_data(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[["ear_mean","ear_std","ear_min","mar_mean","mar_std","pitch_std","perclos"]]
        
        X_train, X_test, y_train, y_test = self.utils.create_train_test_split(df, X.columns.tolist(), 'label')
        
        return X_train, X_test, y_train, y_test
        
    def create_pipeline(self) -> Pipeline:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight='balanced',max_depth=12,min_samples_split=2,min_samples_leaf=1,criterion='gini'))
        ])
        return pipeline

    def save_model(self,pipeline : Pipeline,model_path:str) -> None:
        joblib.dump(pipeline, model_path)
        print('Modelo salvo com sucesso')

    def load_model(self,model_path:str):
        return joblib.load(model_path)