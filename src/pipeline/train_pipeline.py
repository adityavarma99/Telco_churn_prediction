from src.components.data_ingestion import DataIngestion
from src.components.data_tranformation import DataTransformation
from src.components.model_trainer import ModelTrainer

def start_pipeline(file_path):
    try:
        # Step 1: Data Ingestion
        data_ingestion = DataIngestion()
        train_data, test_data = data_ingestion.initiate_data_ingestion(file_path=file_path)
        print(f"Data ingestion completed. Train: {train_data}, Test: {test_data}")

        # Step 2: Data Transformation
        data_transformation = DataTransformation()
        X_train_resampled, y_train_resampled, X_test_selected, y_test, selected_features = data_transformation.initiate_data_transformation(
            train_data, test_data, target_column="Churn"
        )
        print(f"Data transformation completed. Selected features: {selected_features}")

        # Step 3: Model Trainer
        model_trainer = ModelTrainer()
        best_model, best_score = model_trainer.train_and_evaluate(
            X_train_resampled, y_train_resampled, X_test_selected, y_test, selected_features
        )
        print(f"Model training completed. Best Score: {best_score}")

    except Exception as e:
        print(f"Pipeline execution failed: {e}")
