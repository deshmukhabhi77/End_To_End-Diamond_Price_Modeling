import os
import sys
from src.logger import logging
from src.exeption import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_tranformation import DataTransformation

# Initialize the data ingetion configuration

@dataclass
class DataIngetionconfig:
    train_data_path:str = os.path.join('artifacts', 'train_data.csv')
    test_data_path: str = os.path.join('artifacts', 'test_data.csv')
    raw_data_path:str = os.path.join('artifacts','raw_data.csv')

#$ create a class for a data ingetion 

class DataIngetion:
    def __init__(self):
        self.ingetion_config = DataIngetionconfig()
    
    def initiate_data_ingetion(self):
        logging.info('Data Ingetion method starts')

        try:
            df = pd.read_csv(os.path.join('notebooks/data','gamestone.csv'))
            logging.info('dataset raed as pandas data')

            os.makedirs(os.path.dirname(self.ingetion_config.raw_data_path),exist_ok=True)
            logging.info('train test split')

            train_set, test_set = train_test_split(df,test_size=0.30, random_state= 42)

            train_set.to_csv(self.ingetion_config.train_data_path, index = False, header = True)
            test_set.to_csv(self.ingetion_config.test_data_path, index = False, header = True)
            
            logging.info('Ingetion of data is completed')

            return(
                self.ingetion_config.test_data_path,
                self.ingetion_config.train_data_path

            )
        except Exception as e:
            logging.info('Exeception occured where Data ingetion')
            raise CustomException(e,sys)
        