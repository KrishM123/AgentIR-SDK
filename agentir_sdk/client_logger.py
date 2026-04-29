"""
Client-side logging for LangGraph and testing scripts.
Writes logs in spdlog-compatible format to a separate file.
"""
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Default log file location - shared top-level logs folder
DEFAULT_CLIENT_LOG = Path(__file__).resolve().parent.parent / "logs" / "client.log"


class SpdlogFormatter(logging.Formatter):
    """Format logs in spdlog-compatible format."""
    
    def format(self, record):
        # Format: YYYY-MM-DD HH:MM:SS.mmm [LEVEL] [thread] [CATEGORY] message
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S') + f'.{int(record.msecs):03d}'
        level = record.levelname.lower()
        thread = getattr(record, 'thread', 0)
        # Map our logger name to category
        category_map = {
            'client': 'CLIENT',
            'scheduler_client': 'CLIENT',
        }
        category = category_map.get(record.name, 'CLIENT')
        
        return f"{timestamp} [{level:>5}] [{thread:>6}] [{category:>7}] {record.getMessage()}"


def setup_client_logging(log_file: Optional[Path] = None, level=logging.INFO):
    """
    Setup client-side logging to file.
    
    Args:
        log_file: Path to log file. Defaults to logs/client.log
        level: Logging level (default: INFO)
    
    Returns:
        logger instance
    """
    if log_file is None:
        log_file = DEFAULT_CLIENT_LOG
    
    # Ensure log directory exists
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('client')
    logger.setLevel(level)
    
    # Remove any existing handlers to avoid duplicates
    logger.handlers = []
    
    # File handler with spdlog format
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(level)
    file_handler.setFormatter(SpdlogFormatter())
    logger.addHandler(file_handler)
    
    # Also log to console for debugging
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter('[CLIENT] %(message)s'))
    logger.addHandler(console_handler)
    
    return logger


# Global logger instance
_client_logger = None


def get_logger() -> logging.Logger:
    """Get the global client logger."""
    global _client_logger
    if _client_logger is None:
        _client_logger = setup_client_logging()
    return _client_logger


def log_dispatch(api_key: str, rid: int, node: str, prompt_len: int):
    """Log when a request is dispatched to the scheduler."""
    logger = get_logger()
    logger.info(f"Request dispatched: api_key={api_key} rid={rid} node={node} prompt_len={prompt_len}")


def log_response_received(api_key: str, rid: int, node: str, tokens: int, duration_ms: float):
    """Log when a response is received from the scheduler."""
    logger = get_logger()
    logger.info(f"Response received: api_key={api_key} rid={rid} node={node} tokens={tokens} duration={duration_ms:.0f}ms")


if __name__ == "__main__":
    # Test the logging
    setup_client_logging()
    log_dispatch("test-key", 12345, "summarizer", 100)
    log_response_received("test-key", 12345, "summarizer", 50, 1250.5)
    print(f"Test logs written to: {DEFAULT_CLIENT_LOG}")
