import os
import pickle
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.combine import SMOTEENN
from dataclasses import dataclass
import logging

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")
    selected_features_file_path: str = os.path.join("artifacts", "selected_features.pkl")

class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()
        logging.info("DataTransformation initialized with configuration.")

    def get_preprocessor_object(self, numerical_features, categorical_features):
        """
        Creates preprocessing pipelines for numerical and categorical features.
        """
        try:
            logging.info("Creating preprocessing pipelines for numerical and categorical features.")

            # Pipeline for numerical features
            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            # Pipeline for categorical features
            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ])

            # Combined preprocessor
            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_features),
                ("cat_pipeline", cat_pipeline, categorical_features)
            ])

            logging.info("Preprocessing pipelines created successfully.")
            return preprocessor

        except Exception as e:
            logging.error(f"Error in creating preprocessor object: {e}")
            raise e

    def select_top_features(self, X, y, feature_names, k=10):
        """
        Selects the top `k` features using SelectKBest.
        """
        try:
            logging.info(f"Selecting top {k} features using SelectKBest.")

            selector = SelectKBest(score_func=f_classif, k=k)
            X_selected = selector.fit_transform(X, y)
            selected_mask = selector.get_support()
            selected_features = [feature for feature, selected in zip(feature_names, selected_mask) if selected]

            logging.info(f"Top {k} features selected: {selected_features}")

            # Save the selected features
            with open(self.config.selected_features_file_path, "wb") as file:
                pickle.dump(selected_features, file)
            logging.info(f"Selected features saved to {self.config.selected_features_file_path}.")

            return X_selected, selected_features

        except Exception as e:
            logging.error(f"Error in feature selection: {e}")
            raise e

    def apply_smoteenn(self, X, y):
        """
        Applies SMOTEENN to balance the target class distribution.
        """
        try:
            logging.info("Applying SMOTEENN to balance the dataset.")
            smoteenn = SMOTEENN(random_state=42)
            X_resampled, y_resampled = smoteenn.fit_resample(X, y)
            logging.info(f"SMOTEENN applied. Original size: {len(y)}, Resampled size: {len(y_resampled)}.")
            return X_resampled, y_resampled
        except Exception as e:
            logging.error(f"Error applying SMOTEENN: {e}")
            raise e

    def initiate_data_transformation(self, train_path, test_path, target_column):
        """
        Orchestrates the data transformation process: loading data, preprocessing, feature selection, and balancing.
        """
        try:
            logging.info("Starting data transformation process.")

            # Load datasets
            logging.info(f"Loading training data from {train_path} and testing data from {test_path}.")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Separate features and target
            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]
            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]
            logging.info("Separated features and target variable.")

            # Identify numerical and categorical features
            numerical_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
            categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()
            logging.info(f"Numerical features: {numerical_features}")
            logging.info(f"Categorical features: {categorical_features}")

            # Initialize preprocessor
            preprocessor = self.get_preprocessor_object(numerical_features, categorical_features)

            # Fit and transform the training data, transform the testing data
            logging.info("Fitting and transforming the training data.")
            X_train_transformed = preprocessor.fit_transform(X_train)
            logging.info("Transforming the testing data.")
            X_test_transformed = preprocessor.transform(X_test)

            # Get feature names after preprocessing
            transformed_columns = numerical_features + list(preprocessor.named_transformers_["cat_pipeline"].get_feature_names_out(categorical_features))
            logging.info(f"Transformed feature names: {transformed_columns}")

            # Convert transformed data to DataFrame
            X_train_df = pd.DataFrame(X_train_transformed, columns=transformed_columns)
            X_test_df = pd.DataFrame(X_test_transformed, columns=transformed_columns)

            # Feature Selection
            X_train_selected, selected_features = self.select_top_features(X_train_df, y_train, transformed_columns, k=10)
            X_test_selected = X_test_df[selected_features]
            logging.info("Feature selection completed.")

            # Apply SMOTEENN to balance the training dataset
            X_train_resampled, y_train_resampled = self.apply_smoteenn(X_train_selected, y_train)

            # Save the preprocessor for future use
            with open(self.config.preprocessor_obj_file_path, "wb") as file:
                pickle.dump(preprocessor, file)
            logging.info(f"Preprocessor saved to {self.config.preprocessor_obj_file_path}.")

            logging.info("Data transformation process completed successfully.")
            return X_train_resampled, y_train_resampled, X_test_selected, y_test

        except Exception as e:
            logging.error(f"Error in initiate_data_transformation: {e}")
            raise e
