# Import required libraries and modules
import sys  # For system operations and error handling
from dataclasses import dataclass  # For creating configuration classes
import numpy as np  # For numerical operations and array handling
import pandas as pd  # For data manipulation and analysis
import os  # For operating system path operations
from sklearn.compose import ColumnTransformer  # For column-wise transformations
from sklearn.impute import SimpleImputer  # For handling missing values
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # For feature scaling and encoding
from sklearn.pipeline import Pipeline  # For creating processing pipelines
from src.exception import CustomException  # Custom exception class
from src.logger import logging  # Logging module for tracking progress
from src.utils import save_object  # Utility function for saving objects


# Data class for transformation configuration
@dataclass
class DataTransformationConfig:
    # Default path for saving preprocessor object
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

# Main data transformation class
class DataTransformation:
    def __init__(self):
        # Initialize configuration parameters
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        """
        Creates and returns a preprocessing pipeline that handles:
        - Numerical feature processing
        - Categorical feature processing
        """
        try:
            # Define numerical and categorical columns
            numerical_columns = ['writing_score', 'reading_score']
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            # Numerical pipeline for processing numeric features
            num_pipeline = Pipeline(steps=[
                # Replace missing values with median value
                ('imputer', SimpleImputer(strategy='median')),
                # Standardize features by removing mean and scaling to unit variance
                ('scaler', StandardScaler())
            ])

            # Categorical pipeline for processing categorical features
            cat_pipeline = Pipeline(steps=[
                # Replace missing values with most frequent category
                ('imputer', SimpleImputer(strategy='most_frequent')),
                # Convert categorical variables to numerical representations
                ('onehotencoder', OneHotEncoder(handle_unknown='ignore'))
            ])

            # Log column information for monitoring
            logging.info(f"Numerical columns identified: {numerical_columns}")
            logging.info(f"Categorical columns identified: {categorical_columns}")

            # Combine numerical and categorical pipelines using ColumnTransformer
            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_columns),
                ('cat_pipeline', cat_pipeline, categorical_columns)
            ])

            return preprocessor

        except Exception as e:
            # Handle and log any errors in pipeline creation
            logging.error("Error creating preprocessing pipeline")
            raise CustomException(e, sys)

    def initiate_data_tranformation(self, train_path, test_path):
        """
        Executes complete data transformation pipeline:
        1. Load data
        2. Separate features and target
        3. Apply preprocessing
        4. Save preprocessing object
        """
        try:
            # Load training and testing data from CSV files
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Successfully loaded training and testing data")

            # Get preprocessing pipeline object
            logging.info("Initializing preprocessing object")
            preprocessor_obj = self.get_data_transformation_object()

            # Define target column and features
            target_column_name = 'math_score'
            drop_columns = [target_column_name]

            # Separate input features and target variable for training data
            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column_name]

            # Separate input features and target variable for test data
            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = test_df[target_column_name]

            # Apply preprocessing transformations
            logging.info("Applying preprocessing pipeline")
            # Fit and transform on training data
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            # Transform test data using fitted parameters
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            # Combine processed features with target variable
            train_arr = np.c_[
                input_feature_train_arr,  # Processed features
                np.array(target_feature_train_df)  # Target variable
            ]
            test_arr = np.c_[
                input_feature_test_arr,  # Processed features
                np.array(target_feature_test_df)  # Target variable
            ]

            # Save preprocessing pipeline for future use
            logging.info("Saving preprocessing object")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            return (
                train_arr,  # Processed training data array
                test_arr,  # Processed testing data array
                self.data_transformation_config.preprocessor_obj_file_path  # Preprocessor path
            )

        except Exception as e:
            # Handle and log any transformation errors
            logging.error("Error during data transformation process")
            raise CustomException(e, sys)