import pandas as pd 
import numpy as np 
from utils.logger import Logger
from abc import ABC, abstractmethod
from typing import Union

logger = Logger(module_name=__name__, level="DEBUG").get_logger()
class DataStrategy(ABC):
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series,None]:
        pass

class clean_data(DataStrategy):
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        logger.debug("Starting data cleaning process.")
        # Example cleaning steps
        data = data.drop_duplicates()
        logger.debug("Data cleaning completed.")
        return data

