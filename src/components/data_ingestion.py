import os
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "raw_data.csv")
    train_data_path: str = os.path.join("artifacts", "train_data.csv")
    test_data_path: str = os.path.join("artifacts", "test_data.csv")

class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()
        os.makedirs(os.path.dirname(self.config.raw_data_path), exist_ok=True)

    def initiate_data_ingestion(self, file_path: str):
        logging.info("Entered the data ingestion component")
        try:
            logging.info("Starting data ingestion...")
            df = pd.read_csv('notebook/data/Telco-Customer-Churn.csv')
            logging.info(f"Dataset loaded with shape: {df.shape}")

            os.makedirs(os.path.dirname(self.config.train_data_path),exist_ok=True)

            # Save raw data
            df.to_csv(self.config.raw_data_path, index=False,header=True)
            logging.info("Raw data saved successfully.")

            # Split data
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
            logging.info("Split data into training and testing sets.")

            # Save train and test datasets
            train_df.to_csv(self.config.train_data_path, index=False)
            test_df.to_csv(self.config.test_data_path, index=False)
            logging.info("Data ingestion completed successfully.")
            return self.config.train_data_path, self.config.test_data_path

        except Exception as e:
            logging.error(f"Error in data ingestion: {e}")
            raise e
        
if __name__=="__main__":
    obj=DataIngestion()
    obj.initiate_data_ingestion(file_path="notebook/data/Telco-Customer-Churn.csv")
