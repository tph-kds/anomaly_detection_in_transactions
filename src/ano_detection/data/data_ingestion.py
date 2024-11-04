
import os
import sys
import kaggle
# from kaggle.api import (
#     authenticate,
#     competitions_download_files,
#     dataset_download_files, 
#     dataset_metadata, 
#     dataset_list_files, 
#     dataset_list
# )
from kaggle.api.kaggle_api_extended import KaggleApi

from src.ano_detection.logger import logger
from src.ano_detection.exception import MyException
from src.ano_detection.config import DataIngestionConfig
from src.config_params import ROOT_PROJECT

class DataIngestion:
    def __init__(self, 
                 config: DataIngestionConfig, **kwargs):

        super(DataIngestion, self).__init__(**kwargs)
        self.config = config
        self.kaggle_download_dir = ROOT_PROJECT / self.config.download_dir
        self.kaggle_dataset_name = self.config.file_name
        self.kaggle_metadata_name = self.config.metadata_name

        self.api = KaggleApi()

    def __repr__(self):
        return f"{self.__class__.__name__}"

    def initiate_data_ingestion(self):
        try:
            logger.log_message("info", "Initiating data ingestion....")
            self.download_data_from_kaggle()
            self.get_info_of_dataset()
            logger.log_message("info", "Completed data ingestion....")

        except Exception as e:
            logger.log_message("error", f"Error in initiating data ingestion: {e}")
            my_exception = MyException(
                error_message="Error in initiating data ingestion", 
                error_details=sys
            )
            print(my_exception)


    def download_data_from_kaggle(self):

        try:
            logger.log_message("info", "Downloading an original dataset from kaggle website....")

            self.api.authenticate()
            # kaggle.api.competition_download_files(
            #     competition=self.kaggle_dataset_name,
            #     path=self.config.kaggle_download_dir,
            #     force=False,
            #     quiet=True
            # )
            # # kaggle.api.compoe
            ## scrawl data from kaggle and unzip dataset
            self.api.dataset_download_files(
                dataset=self.kaggle_dataset_name,
                path=self.kaggle_download_dir,
                force=False,
                quiet=True,
                unzip=True,
            )
            ## download metadata of below dataset from kaggle
            self.api.dataset_metadata(
                dataset=self.kaggle_dataset_name,
                path=self.kaggle_download_dir,
            )
            

            logger.log_message("info", "Downloaded an original dataset from kaggle website successfully....")

        except Exception as e:
            logger.log_message("error", f"Error in downloading data from kaggle: {e}")
            my_exception = MyException(
                error_message="Error in downloading data from kaggle", 
                error_details=sys
            )
            print(my_exception)
    
    def get_info_of_dataset(self):
        try:
            logger.log_message("info", "Getting info of dataset....")
            
            lis_file = self.api.dataset_list_files(
                dataset = self.kaggle_dataset_name,
                page_size=100,
                page_token=None,
            )
            print(lis_file)


            logger.log_message("info", "Got info of dataset successfully....")

        except Exception as e:
            logger.log_message("error", f"Error in getting info of dataset: {e}")
            my_exception = MyException(
                error_message="Error in getting info of dataset", 
                error_details=sys
            )
            print(my_exception)
            

