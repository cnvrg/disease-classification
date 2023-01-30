import os
import yaml
from tensorflow.keras.models import load_model
import numpy as np
from skimage.transform import resize
from imageio import imread
import base64

class ModelNotFoundError(Exception):
    """Raise if the trained model cannot be found"""

    def __init__(self, model_path):
        super().__init__(model_path)
        self.model_path = model_path

    def __str__(self):
        return f"ModelNotFoundError: The model file does not exist at {self.model_path}. Please check the previous library!"

def validate_path(path):
    """Validates the path to the trained mode
    Args:
        path: path to the trained model file
    Raises:
        ModelNotFoundError: path is not found
    """
    if not os.path.exists(path):
        raise ModelNotFoundError(path)


lib_path = "/disease-classification/"
if os.path.exists("./disease-classification/"):
    lib_path = "./disease-classification/"
validate_path(lib_path + " dcModel.h5")

with open(lib_path + "library.yaml", "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)


def predict(data):

    model = load_model('dcModel.h5')
    decoded = base64.b64decode(data["media"][0])
    arr = np.fromstring(decoded, np.uint8)

    test = np.expand_dims(arr, axis=0)
    result = model.predict(test)
    result = np.argmax(result)

    result_dict = {0: 'Normal', 1: 'Pneumonia'}

    pred = {}    
    pred['Prediction'] = result_dict[result]
    

    return pred