# Import required libraries
import os  # For file path operations
import sys  # For system operations
import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation
import dill  # For object serialization (alternative to pickle)
from src.exception import CustomException  # Custom exception class
from src.logger import logging  # Logging module
from sklearn.metrics import r2_score  # Regression evaluation metric
from sklearn.model_selection import GridSearchCV  # For hyperparameter tuning


def save_object(file_path, obj):
    """
    Save a Python object to a file using dill serialization
    
    Parameters:
    file_path (str): Full path where object will be saved
    obj (any): Python object to be saved
    
    Raises:
    CustomException: If any error occurs during saving
    """
    try:
        # Extract directory path from full file path
        dir_path = os.path.dirname(file_path)
        
        # Create directory structure if it doesn't exist
        # exist_ok=True prevents errors if directory already exists
        os.makedirs(dir_path, exist_ok=True)
        
        # Open file in write-binary mode and serialize object
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)  # Serialize and save object
            
        logging.info(f"Object successfully saved to {file_path}")
            
    except Exception as e:
        # Raise custom exception with error details
        raise CustomException(e, sys)
    

def evaluate_models(X_train, y_train, X_test, y_test, models,param):
    """
    Evaluate multiple machine learning models using R² score
    
    Parameters:
    X_train (array): Training features
    y_train (array): Training target
    X_test (array): Testing features
    y_test (array): Testing target
    models (dict): Dictionary of model objects to evaluate
    
    Returns:
    dict: Dictionary of model names and their R² scores on test data
    
    Raises:
    CustomException: If any error occurs during evaluation
    """
    try:
        report = {}  # Initialize performance report
        
        # Iterate through all provided models
        for i in range(len(list(models))):
            # Get model name and object using index
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]
            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            logging.info(f"Evaluating model: {model_name}")
            
            # Train model on training data
            model.fit(X_train, y_train)
            
            # Generate predictions
            y_train_pred = model.predict(X_train)  # Training predictions
            y_test_pred = model.predict(X_test)    # Testing predictions
            
            # Calculate R² scores
            train_score = r2_score(y_train, y_train_pred)  # Training score
            test_score = r2_score(y_test, y_test_pred)     # Testing score
            
            # Store test score in report
            report[model_name] = test_score
            
            logging.info(f"{model_name} - Train R²: {train_score:.4f}, Test R²: {test_score:.4f}")
            
        return report  # Return all test scores
        
    except Exception as e:
        # Raise custom exception with error details
        raise CustomException(e, sys)
    
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)