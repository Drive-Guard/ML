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
    else:
        print('Configure corretamente o arquivo .env')
        

    
