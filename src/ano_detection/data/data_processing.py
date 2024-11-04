import os
import sys

import pandas as pd

from src.ano_detection.logger import logger
from src.ano_detection.exception import MyException

from src.ano_detection.config import DataProcessingConfig

class DataProcessing:
    def __init__(self, 
                 config: DataProcessingConfig, **kwargs):
        
        super(DataProcessing, self).__init__(**kwargs)
        self.config = config

    def __repr__(self):
        return f"{self.__class__.__name__}"


    def initiate_data_processing(self):
        try:
            logger.log_message("info", "Initiating data processing....")
            self.process_data()
            logger.log_message("info", "Completed data processing....")

        except Exception as e:
            logger.log_message("error", f"Error in initiating data processing: {e}")
            my_exception = MyException(
                error_message="Error in initiating data processing", 
                error_details=sys
            )
            print(my_exception)

    def process_data(self):
        pass

    def show_dataset(self, name_file: str = None) -> pd.DataFrame:
        try:
            logger.log_message("info", "Showing dataset....")
            pd.set_option('display.max_rows', None)
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            pd.set_option('display.max_colwidth', None)
            df = pd.read_csv(name_file)
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

            logger.log_message("info", "Info dataset successfully....")

            return df

        except Exception as e:
            logger.log_message("error", f"Error in info dataset: {e}")
            my_exception = MyException(
                error_message="Error in info dataset", 
                error_details=sys
            )
            print(my_exception)



    
