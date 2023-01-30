import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))import numpy as np

import unittest
import yaml
from tensorflow.keras.models import model, load_model
from train import (train_model, 
IncorrectPathError, 
IncorrectFormatError,)
from yaml.loader import SafeLoader

class TestDiseaseClassification(unittest.TestCase):
    def setUp(self) -> None:
        cfg_path = os.path.dirname(os.path.abspath(__file__))
        cfg_file = cfg_path + "/" + "test_config.yaml"
        self.test_cfg = {}
        with open(cfg_file) as c_info_file:
            self.test_cfg = yaml.load(c_info_file, Loader=SafeLoader)
        self.test_cfg = self.test_cfg[YAML_ARG_TO_TEST]
        self.disease_classification = train_model(
            url=self.test_cfg["lr"],
            token=self.test_cfg["epochs"],
            org=self.test_cfg["batch_size"],
            org=self.test_cfg["model_name"]
        )

        self.accuracy_upper_bound = self.test_cfg["accuracy_upper_bound"]
        self.accuracy_lower_bound = self.test_cfg["accuracy_lower_bound"]


class TrainAccError(TestDiseaseClassification):

    def train_params(self):
        """Checks if the accuracy of trained model is within correct bounds"""
        self.assertTrue(self.accuracy_lower_bound <= self.disease_classification['accuracy'] <= self.accuracy_higher_bound)

    def __str__(self):
        return "TrainAccError: Model train accuracy/loss is not within acceptable range"
