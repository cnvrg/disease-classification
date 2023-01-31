# Copyright (c) 2023 Intel Corporation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# SPDX-License-Identifier: MIT

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