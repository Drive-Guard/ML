from utilitaries.FeatureEngineering import FeatureEngineering
from models.DrowsinessDetectionVoting import DrowsinessDetectionVoting
from models.DrowsinessDetectionNeuralNetwork import DrowsinessDetectionNeuralNetwork

class Orchestrator:
    def __init__(self, model_filepath: str):
        self.feature_engineering = FeatureEngineering()
        self._available_models = {
            'voting': DrowsinessDetectionVoting,
            'neural_network': DrowsinessDetectionNeuralNetwork
        }
        self.models = self.intialize_model_based_on_filepath(model_filepath)

    def intialize_model_based_on_filepath(self, filepath: str) -> list:
        instances = []
        if filepath:
            filepath_lower = filepath.lower()
            for key, model_class in self._available_models.items():
                if key in filepath_lower:
                    instances.append(model_class())
        if not instances:
            print("Aviso: Nenhum modelo específico ('voting' ou 'neural_network') encontrado")
            print("Instanciando TODOS os modelos disponíveis.")
            for model_class in self._available_models.values():
                instances.append(model_class())
        return instances

    def train_models(self, file_path: str, base_model_path: str):
        model = self.models[0]
        print(f"\n--- Treinando {model._model_name} ---")
        model.train_model(file_path, base_model_path)

    def test_models_with_webcam(self, base_model_path: str):
        model = self.models[0]
        print(f"\n--- Iniciando webcam para o: {model._model_name} ---")
        try:
            self.feature_engineering.extract_features_from_webcam(base_model_path)
        except Exception as e:
            print(f"Ocorreu um erro ao usar a webcam com {model._model_name}:", e)

    def extract_features(self, video_path: str):
        self.feature_engineering.extract_features_from_video(video_path)
        print('Extração de features do vídeo finalizada com sucesso')
