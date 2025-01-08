import pandas as pd
import logging
from sklearn.base import BaseEstimator, TransformerMixin
from dataclasses import dataclass

@dataclass
class DataValidationConfig:
    required_columns: list
    numerical_columns: list
    categorical_columns: list

class DataValidator(BaseEstimator, TransformerMixin):
    """
    Validates input data to ensure it meets the expected format, type, and constraints.
    """

    def __init__(self, validation_config: DataValidationConfig):
        self.validation_config = validation_config

    def validate_columns(self, df: pd.DataFrame):
        """
        Validates that all required columns are present in the dataset.
        """
        missing_columns = [
            col for col in self.validation_config.required_columns if col not in df.columns
        ]
        if missing_columns:
            logging.error(f"Missing required columns: {missing_columns}")
            raise ValueError(f"Missing required columns: {missing_columns}")
        logging.info("All required columns are present.")

    def validate_dtypes(self, df: pd.DataFrame):
        """
        Validates that numerical and categorical columns have the correct data types.
        """
        incorrect_numerical = [
            col for col in self.validation_config.numerical_columns
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col])
        ]
        incorrect_categorical = [
            col for col in self.validation_config.categorical_columns
            if col in df.columns and not pd.api.types.is_string_dtype(df[col])
        ]
        if incorrect_numerical:
            logging.error(f"Numerical columns with incorrect data types: {incorrect_numerical}")
            raise TypeError(f"Numerical columns with incorrect data types: {incorrect_numerical}")
        if incorrect_categorical:
            logging.error(f"Categorical columns with incorrect data types: {incorrect_categorical}")
            raise TypeError(f"Categorical columns with incorrect data types: {incorrect_categorical}")
        logging.info("All columns have correct data types.")

    def check_missing_values(self, df: pd.DataFrame):
        """
        Checks for missing values in the dataset.
        """
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            logging.warning(f"Missing values found:\n{missing_values[missing_values > 0]}")
        else:
            logging.info("No missing values found.")

    def fit(self, X, y=None):
        """
        Fit method required for pipeline integration.
        """
        return self

    def transform(self, X: pd.DataFrame):
        """
        Validates the dataset and returns it if all checks pass.
        """
        logging.info("Starting data validation...")
        self.validate_columns(X)
        self.validate_dtypes(X)
        self.check_missing_values(X)
        logging.info("Data validation completed successfully.")
        return X
