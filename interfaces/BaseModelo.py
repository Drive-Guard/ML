from abc import abstractmethod
from utilitaries.Utils import Utils
from utilitaries.FeatureEngineering import FeatureEngineering

class BaseModelo:
    X_train : list
    X_test : list
    y_train : list 
    y_test : list
    def __init__(self) -> None:
        self.utils = Utils()
        self.feature_engineering = FeatureEngineering()
        pass

    @abstractmethod
    def train_model(self):
        """Treina o modelo de machine learning"""
        pass

    @abstractmethod
    def test_model(self):
        """Testa o modelo de machine learning"""
        pass

    @abstractmethod
    def split_data(self):
        """Divide os dados em treino e teste"""
        pass

    @abstractmethod
    def create_pipeline(self):
        """Cria o pipeline de machine learning"""
        pass

    @abstractmethod
    def save_model(self):
        """Salva o modelo de machine learning"""
        pass

    @abstractmethod
    def load_model(self):
        """Carrega o modelo de machine learning"""
        pass