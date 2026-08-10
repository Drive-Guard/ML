from models.DrowsinessDetection import DrowsinessDetection
from utilitaries.FeatureEngineering import FeatureEngineering

def extract_features():
    feature_engineering = FeatureEngineering()
    feature_engineering.extract_features_from_video('data/UTA-RLDD/videos/train')
    print('Finalizado com sucesso')

def orchestrate_model():
    drowsiness_detection = DrowsinessDetection()
    drowsiness_detection.train_model()

if __name__ == "__main__":    
    orchestrate_model()
    #extract_features()
