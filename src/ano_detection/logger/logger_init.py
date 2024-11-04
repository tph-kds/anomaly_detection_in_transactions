import logging
from .logger_setting import MainLoggerHandler
from src.config_params import CONFIG_FILE_PATH
from src.ano_detection.config import ConfiguarationManager
from .grafana_logs_config import GrafanaLogsConfig

logger_config = ConfiguarationManager(CONFIG_FILE_PATH).get_logger_arguments_config()

logFW = GrafanaLogsConfig(
    service_name=logger_config.service_name, 
    instance_id=logger_config.instance_id
)
handler_grafana = logFW.setup_logging(notset=logging.NOTSET)
kwargs = {'handler_grafana': handler_grafana, 'set_grafana_config': False}

logger = MainLoggerHandler(logger_config=logger_config, **kwargs)

