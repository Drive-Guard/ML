from utilitaries.FeatureEngineering import FeatureEngineering
from models.DrowsinessDetectionVoting import DrowsinessDetectionVoting
from models.DrowsinessDetectionNeuralNetwork import DrowsinessDetectionNeuralNetwork
from utilitaries.DriveGuardClient import DriveGuardClient
from utilitaries import CloudPublisher

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
        cliente = DriveGuardClient.do_ambiente()
        if cliente is not None:
            print(f"Enviando leituras para a AWS em {cliente.api_url}")
        try:
            self.feature_engineering.extract_features_from_webcam(base_model_path, cliente)
        except Exception as e:
            print(f"Ocorreu um erro ao usar a webcam com {model._model_name}:", e)
        finally:
            if cliente is not None:
                cliente.finalizar()
                print(f"AWS: {cliente.enviados} leituras enviadas, {cliente.falhas} no buffer offline")

    def extract_features(self, video_path: str):
        self.feature_engineering.extract_features_from_video(video_path)
        print('Extração de features do vídeo finalizada com sucesso')

    def publish_to_cloud(self, model_path: str, features_csv: str = None):
        CloudPublisher.publicar_modelo_s3(model_path)
        if features_csv:
            pipeline = self.feature_engineering.utils.load_model(model_path)
            total = CloudPublisher.publicar_dataset(pipeline, features_csv, self.feature_engineering)
            print(f"Publicacao finalizada: {total} leituras enviadas a API")
