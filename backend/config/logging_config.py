import logging
from logging.config import dictConfig
from .custom_logger import LogConfig
import os


def setup_logging():
    """ Setup logging configuration """

    running_in_fastapi = os.getenv("RUNNING_IN_FASTAPI") == "1"

    if running_in_fastapi:
        logging.getLogger("uvicorn").removeHandler(logging.getLogger("uvicorn").handlers[0])
        dictConfig(LogConfig().dict())
        logger = logging.getLogger("custom_logger")

    else:
        # Default logging configuration for non-FastAPI environments
        logging.basicConfig(
            level=logging.INFO,  # Set the default logging level
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler()  # Output logs to the console
            ]
        )
        logger = logging.getLogger(__name__)

    return logger


logger = setup_logging()
