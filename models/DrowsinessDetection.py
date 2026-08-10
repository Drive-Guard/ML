from utilitaries.Utils import Utils
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import pandas as pd 
from imblearn.over_sampling import SMOTE
from utilitaries.FeatureEngineering import FeatureEngineering
import os
import joblib


class DrowsinessDetection:
    def __init__(self) -> None:
        self.utils = Utils()

    def train_model(self):
        if os.path.exists(f"data/features_data_video.csv"):
            print("Features já foram extraídas")

            df = self.utils.create_dataframe('data/features_data_video.csv')
            X_train, X_test, y_train, y_test = self.split_data(df)
            pipeline = self.create_pipeline()
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            
            self.utils.create_model_evaluation_report(y_test, y_pred , 'Random Forest')
            self.utils.create_confusion_matrix(y_test, y_pred, pipeline.named_steps['rf'],'Random Forest')
            self.save_model()
        else:
            print("Extraindo features...")
            try:
                feature_engineering = FeatureEngineering()
                feature_engineering.extract_features_from_video('data/UTA-RLDD/videos/train')
                print('Finalizado com sucesso')
            except Exception as e:
                print('Erro ao extrair features:', e)
    
    def test_model(self):
        pass

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

    def save_model(self) -> None:
        joblib.dump(self.create_pipeline(), 'models/saved_models/drowsiness_detection_model.pkl')