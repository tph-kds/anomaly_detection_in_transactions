import os
import sys

import pandas as pd
import numpy as np

from src.ano_detection.utils import save_csv
from src.ano_detection.logger import logger
from src.ano_detection.exception import MyException
from src.ano_detection.config import DataProcessingConfig

from src.config_params import ROOT_PROJECT

class DataProcessing:
    def __init__(self, 
                 config: DataProcessingConfig, **kwargs):
        
        super(DataProcessing, self).__init__(**kwargs)
        self.config = config
        self.file_path = self.config.data_path
        self.unuse_features = self.config.unuse_features
        self.root_dir = self.config.root_dir
        self.des_dir = self.config.des_dir

        self.data_processed_path = ROOT_PROJECT / self.root_dir / self.des_dir
        self.path_data = ROOT_PROJECT / self.file_path 

    def __repr__(self):
        return f"{self.__class__.__name__}"


    def initiate_data_processing(self):
        try:
            logger.log_message("info", "Initiating data processing....")
            df, timestamp_df = self.process_data()
            save_csv(df, self.data_processed_path)
            logger.log_message("info", "Completed data processing....")

            return df, timestamp_df

        except Exception as e:
            logger.log_message("error", f"Error in initiating data processing: {e}")
            my_exception = MyException(
                error_message="Error in initiating data processing", 
                error_details=sys
            )
            print(my_exception)

    def process_data(self):
        try:
            logger.log_message("info", "Processing data....")
            df = self.show_dataset(self.path_data)
            self.info_dataset(df)
            df = self.delete_feature(df, self.unuse_features)
            data_processed = df.copy()

            data_processed['timestamp']=pd.to_datetime(data_processed['timestamp'])
            data_processed = data_processed.sort_values('timestamp', ascending=True)
            ### Processing for all features
            timestamp_df =  data_processed["timestamp"]
            data_processed = self.delete_feature(data_processed, ["timestamp"])
            logger.log_message("info", "Processed data successfully....")
            return data_processed, timestamp_df

        except Exception as e:
            logger.log_message("error", f"Error in processing data: {e}")
            my_exception = MyException(
                error_message="Error in processing data", 
                error_details=sys
            )
            print(my_exception)

    def show_dataset(self, file_path: str = None) -> pd.DataFrame:
        try:
            logger.log_message("info", "Showing dataset....")
            pd.set_option('display.max_rows', None)
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            pd.set_option('display.max_colwidth', None)
            df = pd.read_csv(file_path)
            # print(df)
            logger.log_message("info", "Showing dataset successfully....")

            return df

        except Exception as e:
            logger.log_message("error", f"Error in showing dataset: {e}")
            my_exception = MyException(
                error_message="Error in showing dataset", 
                error_details=sys
            )
            print(my_exception)

    def info_dataset(self, df: pd.DataFrame) -> None:
        try:
            logger.log_message("info", "Info dataset....")
            print(f"\nDataset Info: ")
            print(df.info())
            print(f"\nDataset Describe: ")
            print(df.describe())
            print(f"\n Overview of Dataset Null Values: ")
            print(df.isnull().sum())
            print(f"\n Dataset Duplicates: ")
            print(df.duplicated().sum())
            print(f"\n Dataset Unique Values: ")
            print(df.nunique())
            print(f"\n Dataset Head: ")
            print(df.head())

        except Exception as e:
            logger.log_message("error", f"Error in info dataset: {e}")
            my_exception = MyException(
                error_message="Error in info dataset", 
                error_details=sys
            )
            print(my_exception)

    def delete_feature(self, df: pd.DataFrame, features: str) -> pd.DataFrame:
        try:
            logger.log_message("info", "Deleting feature....")
            df = df.drop(features, axis=1)
            logger.log_message("info", "Deleted feature successfully....")

            return df

        except Exception as e:
            logger.log_message("error", f"Error in deleting feature: {e}")
            my_exception = MyException(
                error_message="Error in deleting feature", 
                error_details=sys
            )
            print(my_exception)

    ## Appear some outliers on Q1 and Q3, so we are going to remove them and see the results
    ## We will use 1.5 times the Q1 and 1.5 times the Q3 and focus on amout of transactions in that range
    def remove_outliers(self, numerical_df):
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




    
