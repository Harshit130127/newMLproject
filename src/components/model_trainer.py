# Import required libraries and modules
import os  # For file path operations
import sys  # For system operations and error handling
from dataclasses import dataclass  # For configuration class creation
from catboost import CatBoostRegressor  # CatBoost regression model
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
)  # Ensemble learning models for regression
from sklearn.linear_model import LinearRegression  # Linear regression model
from sklearn.metrics import r2_score  # Evaluation metric for regression
from sklearn.neighbors import KNeighborsRegressor  # KNN regressor
from sklearn.tree import DecisionTreeRegressor  # Decision tree regressor
from xgboost import XGBRegressor  # XGBoost regression model
from src.exception import CustomException  # Custom exception handling
from src.logger import logging  # Logging module
from src.utils import save_object, evaluate_models  # Utility functions

# Configuration class for model training settings
@dataclass
class ModelTrainerConfig:
    # Default path for saving trained model
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

# Main class for model training operations
class ModelTrainer:
    def __init__(self):
        # Initialize configuration parameters
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        """
        Executes complete model training pipeline:
        1. Data preparation
        2. Model evaluation
        3. Best model selection
        4. Model saving
        5. Final evaluation
        """
        try:
            # Split data into features (X) and target (y)
            logging.info("Splitting data into features and targets")
            X_train = train_array[:, :-1]  # All columns except last
            y_train = train_array[:, -1]   # Last column as target
            X_test = test_array[:, :-1]    # All columns except last
            y_test = test_array[:, -1]     # Last column as target

            # Define model dictionary with various regression algorithms
            models = {
                "RandomForestRegressor": RandomForestRegressor(),
                "GradientBoostingRegressor": GradientBoostingRegressor(),
                "LinearRegression": LinearRegression(),
                "DecisionTreeRegressor": DecisionTreeRegressor(),
                "KNeighborsRegressor": KNeighborsRegressor(),
                "CatBoostRegressor": CatBoostRegressor(verbose=0),  # Regression model
                "AdaBoostRegressor": AdaBoostRegressor(),
                "XGBRegressor": XGBRegressor(),                     # Regression model
            }
            params = {
                    "RandomForestRegressor": {  # Key matches model name
                        'n_estimators': [8, 16, 32, 64, 128, 256]
                    },
                    "GradientBoostingRegressor": {  # Key matches model name
                        'learning_rate': [.1, .01, .05, .001],
                        'n_estimators': [8, 16, 32, 64, 128, 256]
                    },
                    "DecisionTreeRegressor": {
                        'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    },
                    "KNeighborsRegressor": {},
                    "LinearRegression": {},
                    "CatBoostRegressor": {
                        'depth': [6, 8, 10],
                        'learning_rate': [0.01, 0.05, 0.1],
                        'iterations': [30, 50, 100]
                    },
                    "AdaBoostRegressor": {
                        'learning_rate': [.1, .01, .05, .001],
                        'n_estimators': [8, 16, 32, 64, 128, 256]
                    },
                    "XGBRegressor": {
                        'learning_rate': [.1, .01, .05, .001],
                        'n_estimators': [8, 16, 32, 64, 128, 256]
                    }
                }
            logging.info("Evaluating candidate models")
            
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params)
            # Evaluate all models using utility function
            logging.info("Evaluating candidate models")
            
            # Identify best performing model
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            # Quality check - minimum acceptable performance
            if best_model_score < 0.6:
                raise CustomException("No suitable model found (score < 0.6)")
            logging.info(f"Best model: {best_model_name} | Score: {best_model_score}")

            # Save best model for future use
            logging.info("Saving best performing model")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Final evaluation on test set
            logging.info("Final model evaluation")
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            
            return r2_square  # Return final evaluation metric

        except Exception as e:
            # Handle and log any training errors
            logging.error("Model training process failed")
            raise CustomException(e, sys)
        
        