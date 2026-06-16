from abc import abstractmethod

class BaseModelo:

    def __init__(self) -> None:
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