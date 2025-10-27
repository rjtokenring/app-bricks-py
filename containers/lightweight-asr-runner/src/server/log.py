import logging
import sys


def load_logger():
    """Load and configure the logger for the Vosk ASR runner."""
    logger = logging.getLogger("lightweight-asr-runner")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
