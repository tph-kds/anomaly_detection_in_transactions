import os
import sys

from abc import ABC, abstractmethod # ( Abstract Base Classes Library )
from src.ano_detection.logger import logger
from src.ano_detection.exception import MyException
from src.ano_detection.config import ModelArgumentsConfig
from src.ano_detection.visualization import VILModel
from src.config_params import ROOT_PROJECT

from sklearn.metrics import (
    accuracy_score, 
    recall_score, 
    precision_score, 
    f1_score,
    roc_curve, 
    auc, 
    precision_recall_curve
)
from sklearn.preprocessing import label_binarize

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
                 config: ModelArgumentsConfig,
                 **kwargs):
        super(BaseModel, self).__init__(**kwargs)   
        self.config = config
        self.root_dir = self.config.root_dir
        self.model_name = self.config.model_name
        self.model_path = ROOT_PROJECT / self.root_dir / self.config.model_path
        self.model_params = self.config.model_params
        self.model_description = self.config.model_description

        self.visualization = VILModel()


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

    @abstractmethod
    def get_score(self, y_true, y_pred, y_scores, n_classes):
        accuracy = accuracy_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')

        y_test_bin = label_binarize(y_true, classes=[0, 1, 2])  # Convert y_test to binary format

        # Compute ROC curve and AUC for each class
        fpr = {}
        tpr = {}
        roc_auc = {}
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_scores[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute Precision-Recall curve and AUC for each class
        precision_curve_pr = {}
        recall_curve_pr = {}
        pr_auc = {}
        for i in range(n_classes):
            precision_curve_pr[i], recall_curve_pr[i], _ = precision_recall_curve(y_test_bin[:, i], y_scores[:, i])
            pr_auc[i] = auc(recall_curve_pr[i], precision_curve_pr[i])


        print(f"Accuracy: {accuracy:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"F1-score: {f1:.2f}")

        self.visualization.auc_roc_curve(fpr, tpr, roc_auc, n_classes)
        self.visualization.auc_pr_curve(precision_curve_pr, recall_curve_pr, pr_auc, n_classes)
        print("Complete!")

        return accuracy, recall, precision, f1


