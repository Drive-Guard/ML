from interfaces.BaseModelo import BaseModelo
from scikeras.wrappers import KerasClassifier
import os
import pandas as pd 
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.model_selection import GroupShuffleSplit
import xgboost as xgb
import tensorflow as tf
class DrowsinessDetection(BaseModelo):
    def __init__(self) -> None:
        super().__init__()
        self.X_test = []
        self.X_train = []
        self.y_test = []
        self.y_train = []

    def train_model(self,file_path:str,model_path:str):
        if os.path.basename(file_path) == 'features_video_train.csv':
            print("Features já foram extraídas")
            df = self.utils.create_dataframe(file_path)
            self.X_train, self.X_test, self.y_train, self.y_test = self.split_data(df)
            # ensemble_model = self.create_ensemble_model()
            # pipeline = self.create_pipeline(ensemble_model,'ensemble')
            neural_network_model = self.create_neural_network_model()
            keras_classififer = KerasClassifier(model=neural_network_model,epochs=500,batch_size=32,verbose=0)
            pipeline = self.create_pipeline(keras_classififer,'neural_network')
            pipeline.fit(self.X_train, self.y_train)
            self.utils.save_model(pipeline,model_path)
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
        feature_columns = ["ear_behavior","pitch_behavior","mar_behavior","perclos"]
        X = df[feature_columns]
        y = df["label"]
        groups = df["participant_id"]

        gfk = GroupShuffleSplit(n_splits=2,test_size=0.3,random_state=42)
        
        for train_idx, test_idx in gfk.split(X, y, groups):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        return X_train, X_test, y_train, y_test

    def create_ensemble_model(self) -> VotingClassifier:
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1,class_weight='balanced',max_depth=10,min_samples_split=4,min_samples_leaf=1,criterion='gini')
        xgb_model = xgb.XGBClassifier(n_estimators=100,max_depth = 3,learning_rate= 0.01,random_state = 42)
        return VotingClassifier(estimators=[('rf',rf_model),('xgb',xgb_model)],voting='hard')
    
    def create_neural_network_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(4,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
        return model
        
    def create_pipeline(self,model,step_name) -> Pipeline:
        pipeline = self.utils.create_pipeline(model,step_name)
        return pipeline