from utilitaries.Utils import Utils
from sklearn.decomposition import PCA
from sklearn.preprocessing import TargetEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import pandas as pd 
from interfaces.BaseModelo import BaseModelo

class AccidentDetection(BaseModelo):
    def __init__(self) -> None:
        super().__init__()
        self.utils = Utils()
        self.components = 4
    
    def train_model(self):        
        df = self.utils.create_dataframe('data\detran2026_trusted.csv')
        X_train, X_test, y_train, y_test = self.split_data(df)
        pipeline = self.create_pipeline()
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        self.utils.create_model_evaluation_report(y_test, y_pred, 'Random Forest')
        self.utils.create_confusion_matrix(y_test, y_pred, pipeline.named_steps['rf'],'Random Forest')

    def split_data(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[['causa_acidente','tipo_acidente','condicao_metereologica','fase_dia','km','br','sentido_via','tracado_reta','tracado_curva','tracado_outros']]
        
        X_train, X_test, y_train, y_test = self.utils.create_train_test_split(df, X.columns.tolist(), 'vitimas')
        
        return X_train, X_test, y_train, y_test
        
    def create_pipeline(self) -> Pipeline:
        pipeline = Pipeline([
            ('target_encoder', TargetEncoder()),
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=self.components)),
            ('rf', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight='balanced',max_depth=8,min_samples_split=5,min_samples_leaf=1))
        ])
        return pipeline