import os
import sys

from abc import ABC, abstractmethod
from src.ano_detection.logger import logger
from src.ano_detection.exception import MyException

class Base_DeepLearningModel(ABC):
    """
    Base class for Deep Learning Model

    Args:
        config: object

    Returns:
        completed all methods (functions) of a deep learning model

    """

    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def forward(self, x_train):
        pass

    @abstractmethod
    def predict(self, x_test):
        pass

    @abstractmethod
    def train_model(self):
        pass

    @abstractmethod
    def validation_model(self):
        pass

    @abstractmethod
    def test_model(self):
        pass


    @abstractmethod
    def save_model(self):
        pass

    @abstractmethod
    def load_model(self):
        pass

