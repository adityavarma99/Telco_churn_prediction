from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV  # Changed from RandomizedSearchCV to GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import pandas as pd

from dataclasses import dataclass

from src.utils import save_object
from src.exception import CustomException
from src.logger import logging


@dataclass
class ModelTrainerConfig:
    trained_model_path: str = os.path.join("artifacts", "best_model.pkl")


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def train_and_evaluate(self, X_train, y_train, X_test, y_test, selected_features):
        try:
            logging.info("Starting model training with hyperparameter tuning")

            # Ensure transformed datasets are converted into DataFrames with proper feature names
            if not isinstance(X_train, pd.DataFrame):
                X_train = pd.DataFrame(X_train, columns=selected_features)
            if not isinstance(X_test, pd.DataFrame):
                X_test = pd.DataFrame(X_test, columns=selected_features)


            # Define models and hyperparameters
            models = {
                "Random Forest": (RandomForestClassifier(), {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [5, 10, 20, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "bootstrap": [True, False]
                }),
                "Gradient Boosting": (GradientBoostingClassifier(), {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2],
                    "max_depth": [3, 5, 7],
                    "min_samples_split": [2, 5],
                    "min_samples_leaf": [1, 2]
                }),
                "Logistic Regression": (LogisticRegression(max_iter=1000, solver="saga"), [
                    {"C": [0.1, 1, 10], "penalty": ["l1"], "solver": ["saga"]},
                    {"C": [0.1, 1, 10], "penalty": ["l2"], "solver": ["saga"]},
                    {"C": [0.1, 1, 10], "penalty": ["elasticnet"], "solver": ["saga"], "l1_ratio": [0.1, 0.5, 0.9]}
                ]),
                "Decision Tree": (DecisionTreeClassifier(), {
                    "criterion": ["gini", "entropy"],
                    "max_depth": [5, 10, 20],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                }),
                "AdaBoost": (AdaBoostClassifier(), {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.05, 0.1, 0.5]
                }),
                "Support Vector Machine": (SVC(), {
                    "C": [0.1, 1, 10, 100],
                    "kernel": ["linear", "rbf", "poly", "sigmoid"],
                    "gamma": ["scale", "auto"],
                    "degree": [2, 3, 4]
                }),
                "K-Nearest Neighbors": (KNeighborsClassifier(), {
                    "n_neighbors": [3, 5, 7, 9],
                    "weights": ["uniform", "distance"],
                    "metric": ["euclidean", "manhattan", "minkowski"]
                })
            }


            best_model = None
            best_score = 0
            best_model_name = None
            best_hyperparameters = None

            for name, (model, params) in models.items():
                logging.info(f"Training model: {name} with hyperparameter tuning...")
                
                # Use GridSearchCV for exhaustive search over parameters
                grid = GridSearchCV(model, params, cv=3, scoring="accuracy", verbose=1)
                grid.fit(X_train, y_train)

                # Evaluate the model
                preds = grid.best_estimator_.predict(X_test)
                acc = accuracy_score(y_test, preds)

                logging.info(f"Model: {name}, Accuracy: {acc}")
                logging.info(f"Classification Report:\n{classification_report(y_test, preds)}")

                # Update the best model if the current model is better
                if acc > best_score:
                    best_model = grid.best_estimator_
                    best_score = acc
                    best_model_name = name
                    best_hyperparameters = grid.best_params_

            # Log details of the best model
            logging.info(f"Best Model: {best_model_name}")
            logging.info(f"Best Hyperparameters: {best_hyperparameters}")
            logging.info(f"Best Model Accuracy: {best_score}")

            # Save the best model using the save_object function
            save_object(self.config.trained_model_path, best_model)
            logging.info(f"Best model saved at: {self.config.trained_model_path}")

            return best_model, best_score

        except Exception as e:
            logging.error(f"Error in model training: {e}")
            raise CustomException(str(e), e)

