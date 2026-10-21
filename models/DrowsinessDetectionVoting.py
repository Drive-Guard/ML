from interfaces.BaseModelo import BaseModelo
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
import xgboost as xgb


class DrowsinessDetectionVoting(BaseModelo):
    def __init__(self) -> None:
        super().__init__()
        self.model_name = 'voting_classifier'
        self.pipeline_step_name = 'voting_classifier'

    def create_model(self) -> VotingClassifier:
        rf_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1,class_weight='balanced',max_depth=12,min_samples_split=10,min_samples_leaf=2,criterion='gini')
        xgb_model = xgb.XGBClassifier(n_estimators=300,max_depth = 6,learning_rate= 0.03,subsample=0.8,colsample_bytree=0.8,num_class=3,objective='multiclass',eval_metric='mlogloss',random_state = 42,n_jobs=-1)
        return VotingClassifier(estimators=[('rf',rf_model),('xgb',xgb_model)],voting='soft')
        