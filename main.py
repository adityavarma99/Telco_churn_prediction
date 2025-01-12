import logging
from src.pipeline.train_pipeline import start_pipeline

def main():
    """
    Main entry point for executing the training pipeline.
    """
    logging.info("Pipeline execution started.")

    # Path to the dataset file
    dataset_path = "notebook/data/Telco-Customer-Churn.csv"

    try:
        # Execute the training pipeline
        start_pipeline(file_path=dataset_path)
        logging.info("Training pipeline executed successfully.")

    except Exception as e:
        logging.error(f"Pipeline execution failed: {e}")
        raise

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("logs/pipeline.log"),  # Log to file
            logging.StreamHandler()                   # Log to console
        ]
    )

    # Run the main function
    main()
