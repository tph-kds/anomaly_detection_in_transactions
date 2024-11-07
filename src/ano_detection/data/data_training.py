import os
import sys
import numpy as np
import pandas as pd

from typing import Optional, List, Union

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    MinMaxScaler, 
    StandardScaler,
)
from sklearn.model_selection import train_test_split


from src.ano_detection.logger import logger
from src.ano_detection.exception import MyException
from src.ano_detection.config import DataTrainingConfig

class DataTraining:

    def __init__(self, 
                 config: DataTrainingConfig, **kwargs):
        super(DataTraining, self).__init__(**kwargs)
        self.config = config
        self.numerical_columns = self.config.numerical_columns # ["transaction_type", "location_region", "purchase_pattern","age_group"]
        self.dtype_convert = self.config.dtype_convert
        self.drop_columns = self.config.drop_columns # ["purchase_pattern_high_value", "purchase_pattern_random"]
        self.RANDOM_SEED = self.config.RANDOM_SEED #2024
        self.TEST_SIZE = self.config.TEST_SIZE #0.2 
        self.VAL_SIZE = self.config.VAL_SIZE #0.15

    def __repr__(self):
        print(f"DataTraining config: {self.config}")
        return f"{self.__class__.__name__}"

    def initiate_data_training(
            self, 
            df: pd.DataFrame) -> Optional[List[Union[str, float, int]]]:
        try:
            logger.log_message("info", "Initiate data training....")
            print("Initiate data training....")
            ## split dataset into X and y == features and target
            X, y = self.split_data(df)
            preprossor = self.create_preprossor(X)

            logger.log_message("info", "Initiate data training successfully....")

            return X, y, preprossor

        except Exception as e:
            logger.log_message("error", f"Error in initiate data training: {e}")
            my_exception = MyException(
                error_message="Error in initiate data training", 
                error_details=sys
            )
            print(my_exception)

    def split_data(self, 
                   df: pd.DataFrame) -> Optional[List[Union[str, float, int]]]:
        try:
            logger.log_message("info", "Split dataset into X and y == features and target....")
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
            X.drop(self.drop_columns, axis=1, inplace=True)
            X = self.convert_numerical_feature(X)
            y.replace({'low_risk': 0, 'moderate_risk' : 1, 'high_risk' : 2 }, inplace=True)
            y = y.astype(self.dtype_convert)

            logger.log_message("info", "Split data successfully....")
            return X, y



        except Exception as e:
            logger.log_message("error", f"Error in split data: {e}")
            my_exception = MyException(
                error_message="Error in split data", 
                error_details=sys
            )
            print(my_exception)

    def convert_numerical_feature(self, 
                                  x: pd.DataFrame) -> Optional[List[Union[str, float, int]]]:
        try:
            logger.log_message("info", "Converting numerical feature....")
            # x_dummies = pd.get_dummies(x, columns=self.numerical_columns, drop_first=True, prefix=self.numerical_columns, dtype=dtype_convert)
                # Identify numerical columns
            numerical_values = x.select_dtypes(exclude=["object", "category"]).columns
            ## remove outliers values
            x_dummies = self.remove_outliers(numerical_values)


            # Define preprocessing for categorical and numerical features
            numerical_transformer = SimpleImputer(strategy="mean")

            
            return x_dummies, numerical_transformer

        except Exception as e:
            logger.log_message("error", f"Error in convert numerical feature: {e}")
            my_exception = MyException(
                error_message="Error in convert numerical feature", 
                error_details=sys
            )
            print(my_exception)
    def convert_categorical_feature(self, 
                                   x: pd.DataFrame) -> Optional[List[Union[str, float, int]]]:
        try:
            logger.log_message("info", "Converting categorical feature....")
            # Identify categorical columns
            categorical_values = x.select_dtypes(include=["object", "category"]).columns

            # Define preprocessing for categorical  features
            categorical_transformer = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]
            )

            return categorical_values, categorical_transformer

        except Exception as e:
            logger.log_message("error", f"Error in convert categorical feature: {e}")
            my_exception = MyException(
                error_message="Error in convert categorical feature", 
                error_details=sys
            )
            print(my_exception)
    
    def remove_outliers(self, 
                        numerical_df: pd.DataFrame) -> Optional[List[Union[str, float, int]]]:
        try:
            logger.log_message("info", "Removing outliers for numerical features....")
            for num_col in numerical_df.columns:

                # Calculate Q1, Q3, and IQR
                q1_value = np.percentile(numerical_df[num_col], 25)
                q3_value = np.percentile(numerical_df[num_col], 75)
                iqr = q3_value - q1_value

                # Define the lower and upper bounds for outliers
                lower_bound = q1_value - 1.5 * iqr
                upper_bound = q3_value + 1.5 * iqr

                # Filter out the outliers and replace with NaN
                numerical_df[num_col] = np.where((numerical_df[num_col] < lower_bound) | (numerical_df[num_col] > upper_bound), np.nan, numerical_df[num_col])

            # Optionally, drop or fill NaN values (outliers removed)
            numerical_df = numerical_df.fillna(method='ffill')  # You can replace this with .fillna(method='ffill') or a different imputation method if desired

            logger.log_message("info", "Removed outliers successfully numerical features....")
            return numerical_df

        except Exception as e:
            logger.log_message("error", f"Error in removing outliers: {e}")
            my_exception = MyException(
                error_message="Error in removing outliers", 
                error_details=sys
            )
            print(my_exception)

    def create_preprossor(self, 
                          X: pd.DataFrame
                          ) -> Optional[List[Union[str, float, int]]]:
        try:
            logger.log_message("info", "Creating preprocessor....")
            numerical_values, numerical_transformer = self.convert_numerical_feature(X)
            categorical_values, categorical_transformer = self.convert_categorical_feature(X)

            # Create a ColumnTransformer for preprocessing
            preprocessor = ColumnTransformer(
                transformers=[
                    ("numerical", numerical_transformer, numerical_values),
                    ("categorical", categorical_transformer, categorical_values),
                ]
            )

            logger.log_message("info", "Created preprocessor successfully....")

            return preprocessor

        except Exception as e:
            logger.log_message("error", f"Error in creating preprocessor: {e}")
            my_exception = MyException(
                error_message="Error in creating preprocessor", 
                error_details=sys
            )
            print(my_exception)

    def create_data_training(self, 
                             X: pd.DataFrame, 
                             y: pd.DataFrame,
                             ) -> Optional[List[Union[str, float, int]]]:
        try:
            logger.log_message("info", "Creating data training....")

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = TEST_SIZE, random_state = RANDOM_SEED)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = VAL_SIZE, random_state = RANDOM_SEED)

            # print(f"X_train shape: {X_train.shape}")
            # print(f"y_train shape: {y_train.shape}")
            # print(f"X_val shape: {X_val.shape}")
            # print(f"y_val shape: {y_val.shape}")
            # print(f"X_test shape: {X_test.shape}")
            # print(f"y_test shape: {y_test.shape}")

            logger.log_message("info", "Created data training successfully....")

            return X_train, y_train, X_val, y_val, X_test, y_test

        except Exception as e:
            logger.log_message("error", f"Error in creating data training: {e}")
            my_exception = MyException(
                error_message="Error in creating data training", 
                error_details=sys
            )
            print(my_exception)

    

    