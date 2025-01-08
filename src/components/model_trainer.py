from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import logging
import os
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_path: str = os.path.join("artifacts", "best_model.pkl")

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        try:
            logging.info("Starting model training with hyperparameter tuning...")

            # Define models and hyperparameters
            models = {
                "Random Forest": (RandomForestClassifier(), {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [5, 10, 20],
                }),
                "Gradient Boosting": (GradientBoostingClassifier(), {
                    "n_estimators": [50, 100, 150],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 5, 7],
                }),
                "Logistic Regression": (LogisticRegression(max_iter=1000), {
                    "C": [0.1, 1, 10],
                }),
                "Decision Tree": (DecisionTreeClassifier(), {
                    "criterion": ["gini", "entropy"],
                    "max_depth": [5, 10, 15],
                }),
                "AdaBoost": (AdaBoostClassifier(), {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.01, 0.1],
                }),
                "XGBoost": (XGBClassifier(use_label_encoder=False, eval_metric="logloss"), {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.01, 0.1],
                    "max_depth": [3, 5, 7],
                }),
            }

            best_model = None
            best_score = 0

            for name, (model, params) in models.items():
                logging.info(f"Training model: {name} with hyperparameter tuning...")
                grid = RandomizedSearchCV(model, params, cv=3, scoring="accuracy", n_iter=10, random_state=42)
                grid.fit(X_train, y_train)

                # Evaluate the model
                preds = grid.best_estimator_.predict(X_test)
                acc = accuracy_score(y_test, preds)

                logging.info(f"Model: {name}, Accuracy: {acc}")
                logging.info(f"Classification Report:\n{classification_report(y_test, preds)}")

                # Update the best model if current model is better
                if acc > best_score:
                    best_model = grid.best_estimator_
                    best_score = acc

            # Save the best model
            with open(self.config.trained_model_path, "wb") as file:
                pickle.dump(best_model, file)
            logging.info(f"Best model saved with accuracy: {best_score}.")

            return best_model, best_score

        except Exception as e:
            logging.error(f"Error in model training: {e}")
            raise e
