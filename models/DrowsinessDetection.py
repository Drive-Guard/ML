from utilitaries.Utils import Utils
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import pandas as pd 
from imblearn.over_sampling import SMOTE
from utilitaries.FeatureEngineering import FeatureEngineering
import os


class DrowsinessDetection:
    def __init__(self) -> None:
        self.utils = Utils()

    def train_model(self):
        if os.path.exists(f"data/features_data_trusted.csv"):
            print("Features já foram extraídas")

            df = self.utils.create_dataframe('data/features_data_trusted.csv')
            X_train, X_test, y_train, y_test = self.split_data(df)
            pipeline = self.create_pipeline()
            
            # Uso de SMOTE para balancear as classes
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(pipeline.named_steps['scaler'].fit_transform(X_train), y_train)
            rf_smote = pipeline.named_steps['rf']
            rf_smote.fit(X_train_resampled, y_train_resampled)
            y_pred = rf_smote.predict((pipeline.named_steps['scaler'].transform(X_test)))

            self.utils.create_model_evaluation_report(y_test, y_pred , 'Random Forest')
            self.utils.create_confusion_matrix(y_test, y_pred, pipeline.named_steps['rf'],'Random Forest')
        else:
            print("Extraindo features...")
            try:
                feature_engineering = FeatureEngineering()
                feature_engineering.extract_features_from_image('data/train')
                print('Finalizado com sucesso')
            except Exception as e:
                print('Erro ao extrair features:', e)
    
    def train_model_adaboost(self):
        df = self.utils.create_dataframe('data/features_data_trusted.csv')
        X_train, X_test, y_train, y_test = self.split_data(df)
        pipeline = self.create_pipeline()
        
        adaboost = AdaBoostClassifier(n_estimators=100, random_state=42)
        # Uso de SMOTE para balancear as classes
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(pipeline.named_steps['scaler'].fit_transform(X_train), y_train)
        adaboost_smote = adaboost
        adaboost_smote.fit(X_train_resampled, y_train_resampled)
        y_pred = adaboost_smote.predict((pipeline.named_steps['scaler'].transform(X_test)))

        self.utils.create_model_evaluation_report(y_test, y_pred , 'Adaboost')
        self.utils.create_confusion_matrix(y_test, y_pred, adaboost,'Adaboost')

    def test_model(self):
        pass

    def split_data(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[['ear','mar','pitch','yaw','roll']]
        
        X_train, X_test, y_train, y_test = self.utils.create_train_test_split(df, X.columns.tolist(), 'label')
        
        return X_train, X_test, y_train, y_test
        
    def create_pipeline(self) -> Pipeline:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced'))
        ])
        return pipeline