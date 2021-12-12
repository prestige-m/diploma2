# -*- coding: utf-8 -*-
import os

from .utils import calc_threshold

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_PATH, "models")
dataset_path = os.path.join(BASE_PATH, "dataset")
classifier_model_path = os.path.join(model_path, "classifier_model.pickle")
label_encoder_path = os.path.join(model_path, "label_encoder.pickle")
