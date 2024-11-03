import logging
import logging_loki

handler = logging_loki.LokiHandler(
 url="https://my-loki-instance/loki/api/v1/push", 
 tags={"application": "my-app"},
 auth=("username", "password"),
 version="1",
)
logger = logging.getLogger("my-logger")
logger.addHandler(handler)
logger.info(
 "Something happened", 
 extra={"tags": {"service": "my-service"}},
)