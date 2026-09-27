from interfaces.BaseModelo import BaseModelo
from scikeras.wrappers import KerasClassifier
import tensorflow as tf

class DrowsinessDetectionNeuralNetwork(BaseModelo):
    def __init__(self) -> None:
        super().__init__()
        self.model_name = 'keras_classifier'
        self.pipeline_step_name = 'keras_classifier'
        
    def create_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(4,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
        return KerasClassifier(model=model,epochs=500,batch_size=32,verbose=0)
    