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

