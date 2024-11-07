import os
import sys

from abc import ABC, abstractmethod # ( Abstract Base Classes Library )
from src.ano_detection.logger import logger
from src.ano_detection.exception import MyException


class BaseModel(ABC):
    """
    Base Model class for all models

    Params:
        config: object

    Returns:
        completed all methods (functions) of a machine leanring model
    
    """
    @abstractmethod
    def __init__(self, 
                 config: object,
                 **kwargs):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def save_model(self):
        pass

    @abstractmethod
    def load_model(self):
        pass


