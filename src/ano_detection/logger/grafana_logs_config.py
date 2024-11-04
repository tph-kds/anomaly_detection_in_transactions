# Import the function to set the global logger provider from the OpenTelemetry logs module.
from opentelemetry._logs import set_logger_provider

# Import the OTLPLogExporter class from the OpenTelemetry gRPC log exporter module.
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter

# Import the LoggerProvider and LoggingHandler classes from the OpenTelemetry SDK logs module.
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler

# Import the BatchLogRecordProcessor class from the OpenTelemetry SDK logs export module.
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor


# Import the Resource class from the OpenTelemetry SDK resources module.
from opentelemetry.sdk.resources import Resource

# Import the logging module.
# import logging
from grpc import StatusCode, RpcError

class GrafanaLogsConfig:
    """
    GrafanaLogsConfig: sets up logging using OpenTelemetry with a specified service name and instance ID.
    """
    
    def __init__(self, service_name, instance_id, endpoint="http://loki:3100/loki/metrics"):
        """
        Initialize the GrafanaLogsConfig: with a service name and instance ID.

        Parameters:
            service_name: Name of the service for logging purposes.
            instance_id: Unique instance ID of the service.

        """
        # Create an instance of LoggerProvider with a Resource object that includes
        # service name and instance ID, identifying the source of the logs.
        self.logger_provider = LoggerProvider(
            resource=Resource.create(
                {
                    "service.name": service_name,
                    "service.instance.id": instance_id,
                }
            )
        )
        self.endpoint = endpoint
        # self.logger_provider = None

    def setup_logging(self, notset):
        """
        Set up the logging configuration.

        Returns:
            LoggingHandler instance configured with the logger provider.
        """
        # try:
        # Set the created LoggerProvider as the global logger provider.
        set_logger_provider(self.logger_provider)

        # Create an instance of OTLPLogExporter with insecure connection.
        exporter = OTLPLogExporter(endpoint="otel-collector:4317", insecure=True)

        # Add a BatchLogRecordProcessor to the logger provider with the exporter.
        self.logger_provider.add_log_record_processor(BatchLogRecordProcessor(exporter))

        # print("Connected to OTLP exporter at", self.endpoint)

        # except RpcError as e:
        #     # Check for UNAVAILABLE status and set up console fallback
        #     if e.code() == StatusCode.UNAVAILABLE:
        #         print(f"Warning: OTLP endpoint '{self.endpoint}' unavailable. Using console logging as fallback.")
        #         self.logger_provider = LoggerProvider(resource=Resource.create(
        #             {
        #                 "service.name": self.service_name,
        #                 "service.instance.id": self.instance_id,
        #             }
        #         ))
        #         set_logger_provider(self.logger_provider)
        #         # logging.getLogger().addHandler(logging.StreamHandler())
        
        # Create a LoggingHandler with the specified logger provider and log level set to NOTSET.
        handler = LoggingHandler(level=notset, logger_provider=self.logger_provider)


        return handler



# from loggingfw import GrafanaLogsConfig

# logFW = GrafanaLogsConfig(service_name='websocket_service', instance_id='1')
# handler = logFW.setup_logging()
# logging.getLogger().addHandler(handler)