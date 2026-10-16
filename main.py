from utilitaries.Orchestrator import Orchestrator
import os
import dotenv

if __name__ == "__main__":  
    dotenv.load_dotenv()
    os.makedirs("data/temp",exist_ok=True)
    behavior = os.environ['MODEL_BEHAVIOR']
    model_filepath = os.environ['MODEL_FILEPATH']
    file_to_train = os.getenv('FILE_TO_TRAIN_MODEL')
    train_video_filepath = os.getenv('MODEL_TRAIN_VIDEO_FILEPATH')

    orchestrator = Orchestrator(model_filepath)
    
    if behavior == 'TEST':
        orchestrator.test_models_with_webcam(model_filepath)
    elif behavior == 'TRAIN': 
        orchestrator.train_models(file_to_train,model_filepath)
    elif behavior == 'EXTRACT':
        orchestrator.extract_features(train_video_filepath)
    elif behavior == 'PUBLISH':
        orchestrator.publish_to_cloud(model_filepath, file_to_train)
    elif behavior == 'TRAIN_PREDICTIVE':
        orchestrator.train_predictive_model(os.getenv('PREDICTIVE_TRAIN_FILE', 'sintetico'),
                                            os.getenv('PREDICTIVE_MODEL_FILEPATH', 'models/saved_models/fatigue_prediction.pkl'))
    else:
        print('Configure corretamente o arquivo .env')
        

    
