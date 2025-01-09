import os
import pickle
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTEN
from dataclasses import dataclass
from src.logger import logging, log_large_data  # Import log_large_data from logger
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")
    selected_features_file_path: str = os.path.join("artifacts", "selected_features.pkl")

class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()
        logging.info("DataTransformation initialized with configuration.")

    def get_preprocessor_object(self, numerical_features, categorical_features):
        try:
            logging.info("Creating preprocessing pipelines for numerical and categorical features.")
            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])
            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ])
            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_features),
                ("cat_pipeline", cat_pipeline, categorical_features)
            ])
            logging.info("Preprocessing pipelines created successfully.")
            return preprocessor
        except Exception as e:
            logging.error(f"Error in creating preprocessor object: {e}")
            raise

    def initiate_data_transformation(self, train_path, test_path, target_column):
        try:
            logging.info("Starting data transformation process.")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            if target_column not in train_df.columns:
                logging.error(f"Target column '{target_column}' not found in training dataset.")
                raise ValueError(f"Target column '{target_column}' not found.")

            X_train = train_df.drop(columns=[target_column], axis=1)
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column], axis=1)
            y_test = test_df[target_column]

            numerical_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
            categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()

            preprocessor = self.get_preprocessor_object(numerical_features, categorical_features)
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            if hasattr(X_train_transformed, 'toarray'):
                X_train_transformed = X_train_transformed.toarray()
                X_test_transformed = X_test_transformed.toarray()

            cat_features = preprocessor.named_transformers_['cat_pipeline']['encoder'].get_feature_names_out(categorical_features)
            transformed_columns = numerical_features + list(cat_features)

            # Log transformed columns using log_large_data
            log_large_data("Transformed columns", transformed_columns)

            # Manual features and mapping
            manual_features = [
                "Dependents_No", "Dependents_Yes", "tenure",
                "OnlineSecurity_No", "OnlineSecurity_Yes", 
                "OnlineBackup_No", "OnlineBackup_Yes",
                "DeviceProtection_No", "DeviceProtection_Yes",
                "TechSupport_No", "TechSupport_Yes",
                "Contract_Month-to-month", "Contract_One year", "Contract_Two year",
                "PaperlessBilling_Yes", "PaperlessBilling_No",
                "MonthlyCharges", "TotalCharges"
            ]
            manual_feature_mapping = {
                "Dependents_No": "Dependents",
                "Dependents_Yes": "Dependents",
                "tenure": "tenure",
                "OnlineSecurity_No": "OnlineSecurity",
                "OnlineSecurity_Yes": "OnlineSecurity",
                "OnlineBackup_No": "OnlineBackup",
                "OnlineBackup_Yes": "OnlineBackup",
                "DeviceProtection_No": "DeviceProtection",
                "DeviceProtection_Yes": "DeviceProtection",
                "TechSupport_No": "TechSupport",
                "TechSupport_Yes": "TechSupport",
                "Contract_Month-to-month": "Contract",
                "Contract_One year": "Contract",
                "Contract_Two year": "Contract",
                "PaperlessBilling_No": "PaperlessBilling",
                "PaperlessBilling_Yes": "PaperlessBilling",
                "MonthlyCharges": "MonthlyCharges",
                "TotalCharges": "TotalCharges"
            }

            selected_features = [
                f for f in transformed_columns if f in manual_features
            ]

            simplified_selected_features = list(set(manual_feature_mapping.get(f, f) for f in selected_features))
            simplified_selected_features.sort()

            # Log simplified final selected features using log_large_data
            log_large_data("Simplified final selected features", simplified_selected_features)

            X_train_df = pd.DataFrame(X_train_transformed, columns=transformed_columns)[selected_features]
            X_test_df = pd.DataFrame(X_test_transformed, columns=transformed_columns)[selected_features]

            save_object(self.config.selected_features_file_path, simplified_selected_features)
            save_object(self.config.preprocessor_obj_file_path, preprocessor)

            smotnee = SMOTEN(random_state=42)
            X_train_resampled, y_train_resampled = smotnee.fit_resample(X_train_df, y_train)

            logging.info("Data transformation process completed successfully.")
            return X_train_resampled, y_train_resampled, X_test_df, y_test, simplified_selected_features

        except Exception as e:
            logging.error(f"Error in data transformation: {e}")
            raise
