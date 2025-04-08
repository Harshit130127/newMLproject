# Import necessary libraries and modules
import os  # For operating system interactions (file paths, directory creation)
import sys  # For system-level operations and error handling
from src.exception import CustomException  # Custom exception class for error handling
from src.logger import logging  # For logging information during execution
import pandas as pd  # For data manipulation and analysis
from sklearn.model_selection import train_test_split  # For splitting data into train/test sets

# Import project components
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer
from dataclasses import dataclass  # For creating data classes with minimal code

# Data class to store configuration paths for data ingestion
@dataclass
class DataIngestionConfig:
    # Define default file paths for processed data using os.path.join for cross-platform compatibility
    train_data_path: str = os.path.join('artifacts', 'train.csv')  # Path to training data
    test_data_path: str = os.path.join('artifacts', 'test.csv')    # Path to testing data
    raw_data_path: str = os.path.join('artifacts', 'data.csv')     # Path to raw/original data

# Class to handle data ingestion process
class DataIngestion:
    def __init__(self):
        # Initialize configuration instance to access file paths
        self.ingestion_config = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        """
        Main method to execute the data ingestion pipeline:
        1. Read source data
        2. Create necessary directories
        3. Save raw data
        4. Split into train/test sets
        5. Save processed data
        """
        logging.info("Entered data ingestion method or component")
        try:
            # Read the source CSV file into a pandas DataFrame
            # Note: Path uses backslash which is Windows-specific - might need adjustment for Linux/Mac
            df = pd.read_csv('notebook\stud.csv')
            logging.info("Successfully read the dataset as a DataFrame")
            
            # Create directory structure if it doesn't exist
            # os.path.dirname() gets the directory path from the full file path
            # exist_ok=True prevents errors if directory already exists
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            # Save the raw data to CSV file
            # index=False prevents adding pandas index column to the file
            # header=True includes column names in the output
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Raw data saved successfully")
            
            # Split the data into training and testing sets
            logging.info("Initiating train-test split")
            # test_size=0.2 means 20% of data used for testing, 80% for training
            # random_state=42 ensures reproducible splits across runs
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            
            # Save training data to CSV
            train_set.to_csv(
                self.ingestion_config.train_data_path,
                index=False,
                header=True
            )
            # Save testing data to CSV
            test_set.to_csv(
                self.ingestion_config.test_data_path,
                index=False,
                header=True
            )
            logging.info("Data ingestion completed successfully")
            
            # Return paths to processed data for next steps in pipeline
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            # Capture and re-raise any exceptions with custom formatting
            logging.error("Error occurred during data ingestion")
            raise CustomException(e, sys)

# Main execution block - runs when script is called directly
if __name__ == "__main__":
    # Create DataIngestion instance and process data
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    
    # Perform data transformation
    data_transformation = DataTransformation()
    # Using transformed data arrays for model training
    train_arr, test_arr, _ = data_transformation.initiate_data_tranformation(train_data, test_data)
    
    # Train and evaluate models
    modeltrainer = ModelTrainer()
    # Print final model evaluation score
    print(f"Model Training Result: {modeltrainer.initiate_model_trainer(train_arr, test_arr)}")