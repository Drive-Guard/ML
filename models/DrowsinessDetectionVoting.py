from interfaces.BaseModelo import BaseModelo
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
import xgboost as xgb

class DrowsinessDetectionVoting(BaseModelo):
    def __init__(self) -> None:
        super().__init__()
        self.model_name = 'voting_classifier'
        self.pipeline_step_name = 'voting_classifier'

    def create_model(self) -> VotingClassifier:
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1,class_weight='balanced',max_depth=10,min_samples_split=4,min_samples_leaf=1,criterion='gini')
        xgb_model = xgb.XGBClassifier(n_estimators=100,max_depth = 3,learning_rate= 0.01,random_state = 42)
        return VotingClassifier(estimators=[('rf',rf_model),('xgb',xgb_model)],voting='hard')
        