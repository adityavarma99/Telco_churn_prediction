import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)
os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# Custom function for truncated logging
def log_large_data(message, data, max_items=10):
    """
    Logs a message with a truncated preview of a large data list or dictionary.
    
    Args:
        message (str): The log message.
        data (list or dict): The data to log.
        max_items (int): Number of items to include in the preview.
    """
    if isinstance(data, (list, dict)):
        preview = data[:max_items] if isinstance(data, list) else dict(list(data.items())[:max_items])
        logging.info(f"{message} (Preview: {preview}... Total items: {len(data)})")
    else:
        logging.info(f"{message}: {data}")

if __name__ == "__main__":
    logging.info("Logging has started")



'''

import logging
import os
from datetime import datetime

LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path=os.path.join(os.getcwd(),"logs",LOG_FILE)
os.makedirs(logs_path,exist_ok=True)

LOG_FILE_PATH=os.path.join(logs_path,LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

if __name__=="__main__":
    logging.info("logging has started")
'''

'''
import logging
import os
from datetime import datetime

def setup_logging():
    """
    Configures the logging settings with datetime in the log file name.
    """
    try:
        # Create logs directory if it doesn't exist
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)

        # Generate log file name with current date and time
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_name = f"app_{current_time}.log"
        log_file_path = os.path.join(log_dir, log_file_name)

        # Configure logging settings
        logging.basicConfig(
            filename=log_file_path,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

        logging.info(f"Logging setup completed. Log file: {log_file_path}")

    except Exception as e:
        raise Exception(f"Error setting up logging: {e}")
'''

