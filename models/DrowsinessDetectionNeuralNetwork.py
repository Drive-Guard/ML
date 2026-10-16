import os
from interfaces.BaseModelo import BaseModelo
from scikeras.wrappers import KerasClassifier
import tensorflow as tf

TRES_CLASSES = int(os.getenv("DRIVEGUARD_CLASSES", "3")) == 3

class DrowsinessDetectionNeuralNetwork(BaseModelo):
    def __init__(self) -> None:
        super().__init__()
        self.model_name = 'keras_classifier'
        self.pipeline_step_name = 'keras_classifier'
        
    def create_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(3,)),
            tf.keras.layers.Dense(32, activation='relu'),
            # 3 classes (alerta / fadiga / sonolento) ou o binário antigo.
            tf.keras.layers.Dense(3, activation='softmax') if TRES_CLASSES else tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        perda = 'sparse_categorical_crossentropy' if TRES_CLASSES else 'binary_crossentropy'
        model.compile(optimizer='sgd', loss=perda, metrics=['accuracy'])
        return KerasClassifier(model=model,epochs=500,batch_size=32,verbose=0)
    