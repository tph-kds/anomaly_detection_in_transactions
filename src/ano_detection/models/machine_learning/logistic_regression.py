import os
import sys

from .base_model import BaseModel
# from src.ano_detection.config import ModelArgumentsConfig
from src.ano_detection.logger import logger
from src.ano_detection.exception import MyException

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from src.ano_detection.utils import save_pickle, load_pickle



class LogisticRegressionModel(BaseModel):
    """
     Configuration for Logistic Regression Model 

    Params:
        config: object

    Returns:
        model for training and prediction tasks
    
    """
    def __init__(self, **kwargs):
        super(LogisticRegressionModel, self).__init__(**kwargs)   

        # self.config = kwargs.get("config")
        
        self.lr_model = LogisticRegression(random_state=2024,
                                    max_iter=1000,
                                    solver='lbfgs',
                                    multi_class='multinomial')

    def __repr__(self):
        print(f"LogisticRegressionModel config: {self.config}")
        logger.log_message("info", f"LLogisticRegressionModel config: {self.config}")
        return f"{self.__class__.__name__}"

    def fit(self, X_train, y_train):
        try:
            logger.log_message("info", "Fitting Logistic Regression Model....")
            self.lr_model.fit(X_train, y_train)

            logger.log_message("info", "Fitted Logistic Regression Model successfully....")

        except Exception as e:
            logger.log_message("error", f"Error in fitting Logistic Regression Model: {e}")
            my_exception = MyException(
                error_message="Error in fitting Logistic Regression Model", 
                error_details=sys
            )
            print(my_exception)

    def predict(self, X_test, y_test):
        try:
            logger.log_message("info", "Predicting Logistic Regression Model....")
            y_pred_gbc = self.lr_model.predict(X_test)
            print(classification_report(y_test, y_pred_gbc))

            logger.log_message("info", "Predicted Logistic Regression Model successfully....")

            return y_pred_gbc
        
        except Exception as e:
            logger.log_message("error", f"Error in predicting Logistic Regression Model: {e}")
            my_exception = MyException(
                error_message="Error in predicting Logistic Regression Model", 
                error_details=sys
            )
            print(my_exception)


    def save_model(self):
        try:
            logger.log_message("info", "Saving Logistic Regression Model....")
            save_pickle(self.lr_model, self.model_path)

            logger.log_message("info", "Saved Logistic Regression Model successfully....")

        except Exception as e:
            logger.log_message("error", f"Error in saving Logistic Regression Model: {e}")
            my_exception = MyException(
                error_message="Error in saving Logistic Regression Model", 
                error_details=sys
            )
            print(my_exception)

    def load_model(self):
        try:
            logger.log_message("info", "Loading Logistic Regression Model....")
            self.lr_model = load_pickle(self.model_path)

            logger.log_message("info", "Loaded Logistic Regression Model successfully....")

        except Exception as e:
            logger.log_message("error", f"Error in loading Logistic Regression Model: {e}")
            my_exception = MyException(
                error_message="Error in loading Logistic Regression Model",
                error_details=sys

            )
            print(my_exception)


