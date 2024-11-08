
from src.ano_detection.utils.config import read_yaml, create_directories
from src.config_params import (
    CONFIG_FILE_PATH,
    # VECTOR_FILE_PATH,
)
from .arguments_config import (
    LoggerArgumentsConfig,
    ExceptionArgumentsConfig,
    DataIngestionConfig,
    DataProcessingConfig,
    DataTrainingConfig,
    ModelArgumentsConfig

)


class ConfiguarationManager:
    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
    ):
        super(ConfiguarationManager, self).__init__()

        self.config = read_yaml(config_filepath)
        create_directories([self.config.artifacts_root])

    def get_logger_arguments_config(self) -> LoggerArgumentsConfig:
        config = self.config.logger

        create_directories([config.root_dir])

        data_logger_config = LoggerArgumentsConfig(
            name=config.name,
            log_dir=config.log_dir,
            name_file_logs=config.name_file_logs,
            format_logging=config.format_logging,
            datefmt_logging=config.datefmt_logging,
            service_name=config.service_name,
            instance_id=config.instance_id,
        )

        return data_logger_config

    def get_exception_arguments_config(self) -> ExceptionArgumentsConfig:
        config = self.config.exception

        # create_directories([config.root_dir])

        data_exception_config = ExceptionArgumentsConfig(
            error_message=config.exception.error_message,
            error_details=config.exception.error_details,
        )

        return data_exception_config
    
    ### DATA INGESTION CONFIG PARAMS PHASE ###
    def get_data_ingestion_arguments_config(self) -> DataIngestionConfig:
        config = self.config.data.stages.ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            download_dir=config.download_dir,
            file_name=config.file_name,
            metadata_name=config.metadata_name,
            target_name = config.target_name

        )

        return data_ingestion_config
    
    ### DATA PROCESSING CONFIG PARAMS PHASE ###
    def get_data_processing_arguments_config(self) -> DataProcessingConfig:
        config = self.config.data.stages.processing

        create_directories([config.root_dir])

        data_processing_config = DataProcessingConfig(
            root_dir=config.root_dir,
            des_dir=config.des_dir,
            data_path= config.data_path,
            unuse_features=config.unuse_features,
        )

        return data_processing_config
    
    ### DATA TRAINING CONFIG PARAMS PHASE ###
    def get_data_training_arguments_config(self) -> DataTrainingConfig:
        config = self.config.data.stages.training

        create_directories([config.root_dir])

        data_training_config = DataTrainingConfig(
            root_dir=config.root_dir,
            des_dir=config.des_dir,
            numerical_columns=config.numerical_columns,
            dtype_convert=config.dtype_convert,
            drop_columns=config.drop_columns,
            RANDOM_SEED = config.RANDOM_SEED,
            TEST_SIZE = config.TEST_SIZE,
            VAL_SIZE = config.VAL_SIZE,

        )

        return data_training_config
    
    ### MODEL TRAINING CONFIG PARAMS PHASE ###
    def get_model_training_arguments_config(self) -> ModelArgumentsConfig:
        config = self.config.model.stages.models

        create_directories([config.root_dir])

        model_training_config = ModelArgumentsConfig(
            root_dir=config.root_dir,
            model_name=config.model_name,
            model_path=config.model_path,
            model_params=config.model_params,
            model_description=config.model_description,
        )

        return model_training_config
    