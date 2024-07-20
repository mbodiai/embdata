import logging
import os
import sys
from pathlib import Path

# from logfire import instrument  # noqa F401
# from logfire.integrations.logging import LogfireLoggingHandler


def init_logger(name: str, level: str = "INFO", use_logfire: bool = True, proprogate: bool = False) -> logging.Logger:
    """Setup logging.

    The following variables take precedence over arguments:
    - LOG_LEVEL
    - USE_LOGFIRE
    """
    level = os.getenv("LOG_LEVEL", level)
    use_logfire = os.getenv("USE_LOGFIRE", use_logfire)

    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=level, format=log_format)

    if Path(name).is_file():
        name = Path(name).stem
    logger = logging.getLogger(name)
    logger.propagate = proprogate

    if level == "DEBUG":
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(log_format)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    # if use_logfire:
    #     logfire_handler = LogfireLoggingHandler()
    #     logger.addHandler(logfire_handler)

    Path("outs").mkdir(exist_ok=True)
    logging.FileHandler(f"outs/{name}.log")
    return logger


global_logger = init_logger("mbodied", level=os.getenv("LOG_LEVEL", "INFO"), use_logfire=os.getenv("USE_LOGFIRE", True))
global_logger.debug("Logger initialized")
Path("outs").mkdir(exist_ok=True)
logging.FileHandler("outs/mbodied.log")
