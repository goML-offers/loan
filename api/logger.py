import logging

def setup_logger(log_file):
    # Create a logger and set the logging level to INFO
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create a file handler to write log messages to the specified log file
    file_handler = logging.FileHandler(log_file)

    # Define the log format, including a timestamp
    log_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(log_format)

    # Add the file handler to the logger
    logger.addHandler(file_handler)


    return logger