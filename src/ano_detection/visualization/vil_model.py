import os
import sys

from src.ano_detection.logger import logger
from src.ano_detection.exception import MyException

import matplotlib.pyplot as plt

class VILModel:
    def __init__(self):
        pass
    def auc_roc_curve(self, fpr, tpr, roc_auc, n_classes):
        # ROC Curve
        plt.subplot(2, 1, 1)
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
        plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.5])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')

        plt.tight_layout()
        plt.show()

    def auc_pr_curve(self, precision, recall, pr_auc, n_classes):
        # Precision-Recall Curve
        plt.subplot(2, 1, 2)
        for i in range(n_classes):
            plt.plot(recall[i], precision[i], lw=2, label=f'Class {i} (AUC = {pr_auc[i]:.2f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.5])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')

        plt.tight_layout()
        plt.show()