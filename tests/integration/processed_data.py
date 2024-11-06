import os
import sys

from src.ano_detection.data import DataProcessing
from src.ano_detection.logger import logger
from src.ano_detection.exception import MyException
from src.ano_detection.config import ConfiguarationManager

def processing():

    try:
        logger.log_message("info", "*"*100)
        logger.log_message("info", "\nTesting data processing phase....".upper())

        config = ConfiguarationManager()

        data_processing_config = config.get_data_processing_arguments_config()

        data_processing = DataProcessing(config=data_processing_config)

        df, timestamp_df = data_processing.initiate_data_processing()

        logger.log_message("info", "\nCompleted testing data processing phase....".upper())
        logger.log_message("info", "*"*100)

        return df, timestamp_df
        

    except Exception as e:
        logger.log_message("error", f"Error in testing data processing phase: {e}")
        my_exception = MyException(
            error_message="Error in testing data processing phase", 
            error_details=sys
        )
        print(my_exception)

if __name__ == "__main__":

    df, timestamp_df = processing()
    print(df.head())
    
