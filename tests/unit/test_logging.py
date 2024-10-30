from src.ano_detection.logger import logger

def test_logger():
    logger.log_message("info", "test")
    assert True

    logger.log_message("warning", "test")
    assert True

    logger.log_message("error", "test")
    assert True

    logger.log_message("critical", "test")
    assert True

    logger.log_message("debug", "test")
    assert True

if __name__ == "__main__":
    test_logger()