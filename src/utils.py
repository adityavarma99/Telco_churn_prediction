import pickle
import logging

def save_object(file_path, obj):
    """
    Saves a Python object to a file using pickle.
    """
    try:
        with open(file_path, "wb") as file:
            pickle.dump(obj, file)
        logging.info(f"Object saved successfully to {file_path}.")
    except Exception as e:
        logging.error(f"Error saving object to {file_path}: {e}")
        raise e

def load_object(file_path):
    """
    Loads a Python object from a file using pickle.
    """
    try:
        with open(file_path, "rb") as file:
            obj = pickle.load(file)
        logging.info(f"Object loaded successfully from {file_path}.")
        return obj
    except Exception as e:
        logging.error(f"Error loading object from {file_path}: {e}")
        raise e
