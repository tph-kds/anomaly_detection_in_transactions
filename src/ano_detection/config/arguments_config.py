import sys
from pathlib import Path
from dataclasses import dataclass, field


# LOGGER PARAMS | EXCEPTION PARAMS
@dataclass(frozen=True)
class LoggerArgumentsConfig:
    """
    Logger arguments config class for the logger.

    """

    name: str = field(
        default="anod_logger", 
        metadata={
            "help": "Name of the logger.",
            "type": "string",
        }
    )
    log_dir: str = field(
        default = "logs", 
        metadata = {
            "help": "Directory to save logs.",
            "type": "string",
        }
    )
    name_file_logs: str = field(
        default="running_logs.log", 
        metadata={
            "help": "Name of the log file.",
            "type": "string",
        }
    )
    format_logging: str = field(
        default="[%(asctime)s - { %(levelname)s } - { %(module)s } - %(message)s]",
        metadata={
            "help": "Format of the log file.",
            "type": "string",
        },
    )
    datefmt_logging: str = field(
        default="%m/%d/%Y %H:%M:%S", 
        metadata={
            "help": "Date format of the log file.",
            "type": "string",
        }
    )
    service_name: str = field(
        default="anod_service", 
        metadata={
            "help": "Name of the service showed on the grafana.",
            "type": "string",
        }
    )
    instance_id: int = field(
        default=1, 
        metadata={
            "help": "Instance ID of the service showed on the grafana.",
            "type": "integer",
        }
    )


@dataclass(frozen=True)
class ExceptionArgumentsConfig:
    error_message: str = field(
        default="Error occured in python script name [{file_name}] line number [{line_number}] error message [{error_message}]",
        metadata={
            "help": "Error message for exception.",
            "type": "string",
        },
    )

    error_details: sys = field(
        default=None, 
        metadata={
            "help": "Error details for exception.",
            "type": "string",
        }
    )

@dataclass(frozen=True)
class DataIngestionConfig:
    
    root_dir: str = field(
        default="data", 
        metadata={
            "help": "Root directory of the dataset.",
            "type": "string",
        }
    )

    download_dir: str = field(
        default="data",
        metadata={
            "help": "Download directory of the dataset.",
            "type": "string",
        }
    )

    file_name: str = field(
        default="dataset",
        metadata={
            "help": "Name of the dataset.",
            "type": "string",
        }
    )

    metadata_name: str = field(
        default="metadata",
        metadata={
            "help": "Name of the metadata of the dataset.",
            "type": "string",
        }
    )

    target_name: str = field(
        default="dataset",
        metadata={
            "help": "Name of the target of the dataset.",
            "type": "string",
        }
    )
@dataclass(frozen=True)
class DataProcessingConfig:
    root_dir: str = field(
        default="src/artifacts/data", 
        metadata={
            "help": "Root directory of the dataset.",
            "type": "string",
        }
    )

    des_dir: str = field(
        default="processed/",
        metadata={
            "help": "Download directory of the dataset.",
            "type": "string",
        }
    )

    data_path: str = field(
        default="data/dataset.csv",
        metadata={
            "help": "Path of the dataset.",
            "type": "string",
        }
    )

    unuse_features: list = field(
        default_factory=list,
        # default=["Unnamed: 0", "sending_address", "receiving_address", "ip_prefix"],
        metadata={
            "help": "List of features to be removed.",
            "type": "list",
        }
    )

@dataclass(frozen=True)
class DataTrainingConfig:

    root_dir: str = field(
        default="src/artifacts/data",
        metadata={
            "help": "Root directory of the dataset.",
            "type": "string",
        }
    )
    des_dir: str = field(
        default="pretraining/",
        metadata={
            "help": "Download directory of the dataset.",
            "type": "string",
        }
    ) 
    numerical_columns: list = field(
        default_factory=list,
        # default=["transaction_type", "location_region", "purchase_pattern","age_group"],
        metadata={
            "help": "List of numerical features.",
            "type": "list",
        }
    )

    dtype_convert: str = field(
        default="int64",
        metadata={
            "help": "Data type of the numerical features.",
            "type": "string",
        }
    ) 

    drop_columns: list = field(
        default_factory=list,
        # default=["Unnamed: 0", "sending_address", "receiving_address", "ip_prefix"],
        metadata={
            "help": "List of features to be removed.",
            "type": "list",
        }
    )

    RANDOM_SEED: int = field(
        default=2024, 
        metadata={
            "help": "Random seed for reproducibility.",
            "type": "integer",
        }
    )
    TEST_SIZE: float = field(
        default=0.2, 
        metadata={
            "help": "Test size of the dataset when splitting dataset for training phase.",
            "type": "float",
        }
    ) 
    VAL_SIZE: float = field(
        default=0.15, 
        metadata={
            "help": "Validation size of the dataset when splitting dataset for training phase.",
            "type": "float",
        }
    )

@dataclass(frozen=True)
class ModelArgumentsConfig:
    root_dir: str = field(
        default="src/artifacts/models",
        metadata={
            "help": "Root directory of the dataset.",
            "type": "string",
        }
    )
    model_name: str = field(
        default="name_model",
        metadata={
            "help": "Name of the model.",
            "type": "string",
        }
    )
    model_path: str = field(
        default="model.pkl",
        metadata={
            "help": "Path of the model.",
            "type": "string",
        }
    )
    model_params: dict = field(
        default={"n_estimators": 100, "max_depth": 5, "random_state": 42},
        metadata={
            "help": "Parameters of the model.",
            "type": "dict",
        }
    )
    model_description: str = field(
        default="Model Description",
        metadata={
            "help": "Description of the model.",
            "type": "string",
        }
    )




