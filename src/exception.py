import sys
import traceback

class CustomException(Exception):
    def __init__(self, error_message: str, error_details: Exception):
        super().__init__(error_message)
        self.error_message = self.get_detailed_error_message(error_message, error_details)

    @staticmethod
    def get_detailed_error_message(error_message: str, error_details: Exception):
        _, _, exc_tb = sys.exc_info()
        if exc_tb is not None:
            tb = traceback.extract_tb(exc_tb)
            error_detail = "".join(traceback.format_list(tb))
        else:
            error_detail = "Traceback not available."

        return f"{error_message}\nError Details:\n{error_detail}"
