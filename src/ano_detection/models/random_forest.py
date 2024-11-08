import os
import sys

from .base_model import BaseModel
# from src.ano_detection.config import ModelArgumentsConfig
from src.ano_detection.logger import logger
from src.ano_detection.exception import MyException

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from src.ano_detection.utils import save_pickle, load_pickle



class RandomForestModel(BaseModel):
    """
     Configuration for Random Forest Model 

    Params:
        config: object

    Returns:
        model for training and prediction tasks
    
    """
    def __init__(self, **kwargs):
        super(RandomForestModel, self).__init__(**kwargs)   

        # self.config = kwargs.get("config")
        
        self.random_forest_model = RandomForestClassifier(random_state=2024, 
                                   n_estimators=100, 
                                   criterion="entropy", 
                                   max_depth=10, 
                                   min_samples_split=2,
                                   min_samples_leaf=1,
                                   min_weight_fraction_leaf=0.0,
                                   max_features=1.0)

    def __repr__(self):
        print(f"RandomForestModel config: {self.config}")
        logger.log_message("info", f"RandomForestModel config: {self.config}")
        return f"{self.__class__.__name__}"

    def fit(self, X_train, y_train):
        try:
            logger.log_message("info", "Fitting Random Forest Model....")
            self.random_forest_model.fit(X_train, y_train)

            logger.log_message("info", "Fitted Random Forest Model successfully....")

        except Exception as e:
            logger.log_message("error", f"Error in fitting Random Forest Model: {e}")
            my_exception = MyException(
                error_message="Error in fitting Random Forest Model", 
                error_details=sys
            )
            print(my_exception)

    def predict(self, X_test, y_test):
        try:
            logger.log_message("info", "Predicting Random Forest Model....")
            y_pred_gbc = self.random_forest_model.predict(X_test)
            print(classification_report(y_test, y_pred_gbc))

            logger.log_message("info", "Predicted Random Forest Model successfully....")

            return y_pred_gbc
        
        except Exception as e:
            logger.log_message("error", f"Error in predicting Random Forest Model: {e}")
            my_exception = MyException(
                error_message="Error in predicting Random Forest Model", 
                error_details=sys
            )
            print(my_exception)


    def save_model(self):
        try:
            logger.log_message("info", "Saving Random Forest Model....")
            save_pickle(self.random_forest_model, self.model_path)

            logger.log_message("info", "Saved Random Forest Model successfully....")

        except Exception as e:
            logger.log_message("error", f"Error in saving Random Forest Model: {e}")
            my_exception = MyException(
                error_message="Error in saving Random Forest Model", 
                error_details=sys
            )
            print(my_exception)

    def load_model(self):
        try:
            logger.log_message("info", "Loading Random Forest Model....")
            self.random_forest_model = load_pickle(self.model_path)

            logger.log_message("info", "Loaded Random Forest Model successfully....")

        except Exception as e:
            logger.log_message("error", f"Error in loading Random Forest Model: {e}")
            my_exception = MyException(
                error_message="Error in loading Random Forest Model",
                error_details=sys

            )
            print(my_exception)


