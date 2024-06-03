import logging
import os

def setup_logger(name, log_file, level=logging.INFO):
    if not os.path.exists('logs'):
        os.makedirs('logs')

    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger