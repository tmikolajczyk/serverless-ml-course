
import joblib
import os

class Predict(object):
    
    def __init__(self):
        # NOTE: env var ARTIFACT_FILES_PATH has the local path to the model artifact files        
        self.model = joblib.load(os.environ["ARTIFACT_FILES_PATH"] + "/knn_iris_model.pkl")


    def predict(self, inputs):
        """ Serves a prediction request from a trained model"""
        return self.model.predict(inputs).tolist()
