from utilitaries.Utils import Utils
from models.AccidentDetection import AccidentDetection
from models.DrowsinessDetection import DrowsinessDetection
import pandas as pd

def start_training_models():
    # accident_detection = AccidentDetection()
    # accident_detection.train_model()
    drowsiness_detection = DrowsinessDetection()
    drowsiness_detection.train_model()
    drowsiness_detection.train_model_adaboost()

if __name__ == "__main__":    
    start_training_models()
