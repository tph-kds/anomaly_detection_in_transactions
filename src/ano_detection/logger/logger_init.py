from .trim_rag.logger.logger_setting import MainLoggerHandler
from .trim_rag.config import ConfiguarationManager
from .config_params import CONFIG_FILE_PATH

logger_config = ConfiguarationManager(CONFIG_FILE_PATH).get_logger_arguments_config()

logger = MainLoggerHandler(logger_config=logger_config)
