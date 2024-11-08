import os
import sys

from .base_model import BaseModel
# from src.ano_detection.config import ModelArgumentsConfig
from src.ano_detection.logger import logger
from src.ano_detection.exception import MyException

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report

from src.ano_detection.utils import save_pickle, load_pickle



class GradientBoostingModel(BaseModel):
    """
     Configuration for Gradient Boosting Model 

    Params:
        config: object

    Returns:
        model for training and prediction tasks
    
    """
    def __init__(self, **kwargs):
        super(GradientBoostingModel, self).__init__(**kwargs)   

        # self.config = kwargs.get("config")
        
        self.gradient_boosting_model = GradientBoostingClassifier(random_state=2024,
                                                n_estimators=100,
                                                learning_rate=0.1,
                                                max_depth=10,
                                                min_samples_leaf=1,
                                                min_weight_fraction_leaf=0.0,
                                                max_features=1.0)

    def __repr__(self):
        print(f"GradientBoostingModel config: {self.config}")
        logger.log_message("info", f"GradientBoostingModel config: {self.config}")
        return f"{self.__class__.__name__}"

    def fit(self, X_train, y_train):
        try:
            logger.log_message("info", "Fitting Gradient Boosting Model....")
            self.gradient_boosting_model.fit(X_train, y_train)

            logger.log_message("info", "Fitted Gradient Boosting Model successfully....")

        except Exception as e:
            logger.log_message("error", f"Error in fitting Gradient Boosting Model: {e}")
            my_exception = MyException(
                error_message="Error in fitting Gradient Boosting Model", 
                error_details=sys
            )
            print(my_exception)

    def predict(self, X_test, y_test):
        try:
            logger.log_message("info", "Predicting Gradient Boosting Model....")
            y_pred_gbc = self.gradient_boosting_model.predict(X_test)
            print(classification_report(y_test, y_pred_gbc))

            logger.log_message("info", "Predicted Gradient Boosting Model successfully....")

            return y_pred_gbc
        
        except Exception as e:
            logger.log_message("error", f"Error in predicting Gradient Boosting Model: {e}")
            my_exception = MyException(
                error_message="Error in predicting Gradient Boosting Model", 
                error_details=sys
            )
            print(my_exception)


    def save_model(self):
        try:
            logger.log_message("info", "Saving Gradient Boosting Model....")
            save_pickle(self.gradient_boosting_model, self.model_path)

            logger.log_message("info", "Saved Gradient Boosting Model successfully....")

        except Exception as e:
            logger.log_message("error", f"Error in saving Gradient Boosting Model: {e}")
            my_exception = MyException(
                error_message="Error in saving Gradient Boosting Model", 
                error_details=sys
            )
            print(my_exception)

    def load_model(self):
        try:
            logger.log_message("info", "Loading Gradient Boosting Model....")
            self.gradient_boosting_model = load_pickle(self.model_path)

            logger.log_message("info", "Loaded Gradient Boosting Model successfully....")

        except Exception as e:
            logger.log_message("error", f"Error in loading Gradient Boosting Model: {e}")
            my_exception = MyException(
                error_message="Error in loading Gradient Boosting Model",
                error_details=sys

            )
            print(my_exception)


