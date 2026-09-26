from models.DrowsinessDetection import DrowsinessDetection
from utilitaries.FeatureEngineering import FeatureEngineering
import os
import dotenv

def extract_features(feature_engineering : FeatureEngineering,feature_to_extract : str):
    feature_engineering.extract_features_from_video(feature_to_extract)
    print('Extração de features do vídeo finalizado com sucesso')

def capture_webcam(feature_engineering : FeatureEngineering, drowsiness_model : DrowsinessDetection, model_to_use : str):
    try:
        feature_engineering.extract_features_from_webcam(model_to_use)
    except Exception as e:
        print("Ocorreu um erro ao usar a webcam:", e)

def orchestrate_model(drowsiness_model: DrowsinessDetection,file_path : str,model_path : str):
    drowsiness_model.train_model(file_path,model_path)

if __name__ == "__main__":  
    dotenv.load_dotenv()
    os.makedirs("data/temp",exist_ok=True)
    feature_extraction = FeatureEngineering()
    drowsiness_model = DrowsinessDetection()

    behavior = os.environ['MODEL_BEHAVIOR']
    if behavior == 'TEST':
        capture_webcam(feature_extraction, drowsiness_model, os.getenv('MODEL_FILEPATH'))
    elif behavior == 'TRAIN':
        orchestrate_model(drowsiness_model,os.getenv('FILE_TO_TRAIN_MODEL'),os.getenv('MODEL_FILEPATH'))
    elif behavior == 'EXTRACT_FEATURES':
        extract_features(feature_extraction,os.getenv('MODEL_TRAIN_VIDEO_FILEPATH'))
    else:
        print('Configure corretamente o arquivo .env')
        

    
