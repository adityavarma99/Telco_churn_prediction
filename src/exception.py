import sys
import logging

class CustomException(Exception):
    """
    Custom exception class for better error handling and logging.
    """
    def __init__(self, error_message: str, error_details: sys):
        super().__init__(error_message)
        self.error_message = CustomException.get_detailed_error_message(error_message, error_details)

    @staticmethod
    def get_detailed_error_message(error_message: str, error_details: sys) -> str:
        """
        Constructs a detailed error message.
        """
        _, _, exc_tb = error_details.exc_info()
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        detailed_message = f"Error occurred in file [{file_name}] at line [{line_number}]: {error_message}"
        return detailed_message

    def __str__(self):
        return self.error_message
