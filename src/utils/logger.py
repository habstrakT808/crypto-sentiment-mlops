"""
Logging Configuration
Centralized logging setup for the application
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from pythonjsonlogger import jsonlogger

from src.utils.config import Config

def setup_logger(
    name: str,
    log_file: str = None,
    level: str = None
) -> logging.Logger:
    """
    Setup logger with both file and console handlers
    
    Args:
        name: Logger name
        log_file: Optional log file name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    
    # Set level
    log_level = getattr(logging, level or Config.LOG_LEVEL)
    logger.setLevel(log_level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    json_formatter = jsonlogger.JsonFormatter(
        '%(asctime)s %(name)s %(levelname)s %(funcName)s %(lineno)d %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(detailed_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Config.LOGS_DIR / log_file
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(json_formatter)
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger

# Create default application logger
app_logger = setup_logger(
    "crypto_sentiment",
    log_file=f"app_{datetime.now().strftime('%Y%m%d')}.log"
)