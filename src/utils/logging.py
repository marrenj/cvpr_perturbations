import logging
import os
import sys


def setup_logger(log_file_path):
    """
    Set up logger to write to both console and file.
    
    Args:
        log_file_path (str): Path to the log file
    
    Returns:
        logger: Configured logger object
    """
    # Create logger
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers to avoid duplicates
    logger.handlers = []
    
    # Create formatters
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                                datefmt='%Y-%m-%d %H:%M:%S')
    
    # File handler
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    file_handler = logging.FileHandler(log_file_path, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info("="*80)
    logger.info("Starting Process")
    logger.info(f"Log file: {log_file_path}")
    logger.info("="*80)
    
    return logger