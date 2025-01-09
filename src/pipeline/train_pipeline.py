from src.components.data_ingestion import DataIngestion
from src.components.data_tranformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging

def start_pipeline(file_path):
    """
    Executes the complete training pipeline: data ingestion, transformation, and model training.
    """
    try:
        logging.info("Starting the training pipeline...")

        # Step 1: Data Ingestion
        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion(file_path)
        logging.info(f"Data ingestion completed. Train data: {train_path}, Test data: {test_path}")

        # Step 2: Data Transformation
        transformation = DataTransformation()
        X_train, y_train, X_test, y_test, selected_features = transformation.initiate_data_transformation(
            train_path, test_path, target_column="Churn"
        )
        logging.info(f"Data transformation completed. Training and testing data prepared.")

        # Step 3: Model Training
        trainer = ModelTrainer()
        best_model, best_score = trainer.train_and_evaluate(X_train, y_train, X_test, y_test, selected_features)
        logging.info(f"Model training completed. Best Model Accuracy: {best_score}")

        return best_model, best_score

    except Exception as e:
        logging.error(f"Error in training pipeline: {e}")
        raise e
