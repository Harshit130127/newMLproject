import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split

from src.components.data_transformation import DataTransformation

from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig

from src.components.model_trainer import ModelTrainer

from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

    
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info("entered data ingestion method or component")
        try:
            df=pd.read_csv('notebook\stud.csv')
            logging.info("read the dataset as dataframe")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            
            logging.info("ingestion of data is completed")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
                
            )
        except Exception as e:
            raise CustomException(e,sys)
            
        
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()
    
    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_tranformation(train_data,test_data)
    
    
    
    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))
    
    
# Why and When We Use This Code
# Purpose: This code is used in machine learning projects to prepare data for training and testing models. Properly preparing data is crucial because the quality of the data directly affects the performance of the model.

# When to Use: You would use this code when you have a dataset that you want to split into training and testing sets. This is typically done before training a machine learning model to ensure that the model can generalize well to new, unseen data.
    
    
    
    
    
    '''def initiate_data_ingestion(self):: This method is where the main work happens. It reads the data, processes it, and saves it in the correct format.

logging.info("entered data ingestion method or component"): This logs a message indicating that the data ingestion process has started. It's useful for tracking the program's progress.

try:: This starts a block of code that will attempt to run. If an error occurs, it will jump to the except block.

df=pd.read_csv('notebook\stud.csv'): This line reads a CSV file named stud.csv located in the notebook folder and loads it into a pandas DataFrame called df. A DataFrame is like a table where we can easily manipulate data.

logging.info("read the dataset as dataframe"): This logs a message indicating that the dataset has been successfully read into a DataFrame.

os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True): This line creates the directory (folder) where we will save our data files if it doesn't already exist. The exist_ok=True argument means that it won't raise an error if the folder already exists.

df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True): This saves the original DataFrame df to a CSV file at the path specified by raw_data_path. The index=False argument means we don't want to save the row numbers, and header=True means we want to include the column names.

logging.info("train test split initiated"): This logs a message indicating that the process of splitting the data into training and testing sets is starting.

train_set,test_set=train_test_split(df,test_size=0.2,random_state=42): This line splits the DataFrame df into two parts:

train_set: This will contain 80% of the data, which we will use to train our model.
test_set: This will contain 20% of the data, which we will use to test how well our model performs. The random_state=42 ensures that the split is the same every time we run the code, which is important for consistency.
train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True): This saves the training set to a CSV file at the path specified by train_data_path.

test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True): This saves the testing set to a CSV file at the path specified by test_data_path.

logging.info("ingestion of data is completed"): This logs a message indicating that the data ingestion process has finished successfully.

return (...): This returns the paths of the training and testing data files, which can be used later in the program.

except Exception as e:: If any error occurs during the try block, this line catches the error.

raise CustomException(e,sys): This raises a custom exception with the error message and system information, allowing us to handle the error in a specific way.'''