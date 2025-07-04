import logging
import os


def create_logger(log_dir, log_name):
    """
    Create a log file named log_name under the directory log_dir.
    :param log_dir: Directory to save the log file
    :param log_name: Name of the log file (without extension)
    :return: Logger object
    """
    # Create logger object
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # Set log format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d')
    # Create file handler in 'w' mode to overwrite existing file ('a' mode appends)
    file_handler = logging.FileHandler(os.path.join(log_dir, f"{log_name}.log"), mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    # Add handler to logger
    logger.addHandler(file_handler)
    return logger


def log_string(logger, str_print):
    """
    Print log message.
    :param logger: Logger object
    :param str_print: Message string to log and print
    :return: None
    """
    logger.info(str_print)
    print(str_print)
