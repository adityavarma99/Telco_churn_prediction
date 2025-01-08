import os
import pickle
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTEN  # Changed from SMOTE to SMOTNEE
from dataclasses import dataclass
from src.logger import logging
from src.utils import save_object  # Corrected import

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

            if X_train_transformed.shape[1] == 0:
                logging.error("No features generated after transformation.")
                raise ValueError("No features generated after transformation.")

            # Debugging and validation logs
            logging.info(f"Shape of X_train_transformed: {X_train_transformed.shape}")
            logging.info(f"Number of numerical features: {len(numerical_features)}")
            logging.info(f"Number of categorical features: {len(categorical_features)}")

            cat_features = preprocessor.named_transformers_['cat_pipeline']['encoder'].get_feature_names_out(categorical_features)
            logging.info(f"Extracted categorical features: {len(cat_features)}")

            transformed_columns = numerical_features + list(cat_features)
            logging.info(f"Total transformed columns: {len(transformed_columns)}")

            if len(transformed_columns) != X_train_transformed.shape[1]:
                logging.error(
                    f"Mismatch: {len(transformed_columns)} column names vs {X_train_transformed.shape[1]} transformed columns."
                )
                raise ValueError("Mismatch between transformed data columns and column names.")

            # Ensure dense representation if sparse
            if hasattr(X_train_transformed, 'toarray'):
                X_train_transformed = X_train_transformed.toarray()
                X_test_transformed = X_test_transformed.toarray()

            X_train_df = pd.DataFrame(X_train_transformed, columns=transformed_columns)
            X_test_df = pd.DataFrame(X_test_transformed, columns=transformed_columns)

            selector = SelectKBest(score_func=f_classif, k=10)
            X_train_selected = selector.fit_transform(X_train_df, y_train)
            selected_mask = selector.get_support()
            selected_features = [feature for feature, selected in zip(transformed_columns, selected_mask) if selected]

            if not selected_features:
                logging.error("No features selected by SelectKBest.")
                raise ValueError("No features selected by SelectKBest.")

            logging.info(f"Selected features: {selected_features}")
            X_test_selected = X_test_df[selected_features]

            save_object(self.config.selected_features_file_path, selected_features)

            # Replacing SMOTE with SMOTNEE for better quality sampling
            smotnee = SMOTEN(random_state=42)
            X_train_resampled, y_train_resampled = smotnee.fit_resample(X_train_selected, y_train)

            save_object(self.config.preprocessor_obj_file_path, preprocessor)

            logging.info("Data transformation process completed successfully.")
            return X_train_resampled, y_train_resampled, X_test_selected, y_test, selected_features
        except Exception as e:
            logging.error(f"Error in data transformation: {e}")
            raise
