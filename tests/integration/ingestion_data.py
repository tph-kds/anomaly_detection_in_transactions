import os
import sys

from src.ano_detection.data import DataIngestion
from src.ano_detection.logger import logger
from src.ano_detection.exception import MyException
from src.ano_detection.config import ConfiguarationManager

def ingestion():

    try:
        logger.log_message("info", "Testing data ingestion phase....")

        config = ConfiguarationManager()

        data_ingestion_config = config.get_data_ingestion_arguments_config()

        data_ingestion = DataIngestion(config=data_ingestion_config)

        data_ingestion.initiate_data_ingestion()

        logger.log_message("info", "Completed testing data ingestion phase....")

    except Exception as e:
        logger.log_message("error", f"Error in testing data ingestion phase: {e}")
        my_exception = MyException(
            error_message="Error in testing data ingestion phase", 
            error_details=sys
        )
        print(my_exception)

if __name__ == "__main__":

    ingestion()
    
