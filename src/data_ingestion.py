
import pandas as pd 
from utils.logger import Logger
from sqlalchemy import create_engine
from dotenv import load_dotenv

from sqlalchemy.engine import Engine
import os
from sklearn.model_selection import train_test_split
from typing import Tuple, Union
from utils.yaml_loader import load_yaml

logger = Logger("data_ingestion", level='DEBUG').get_logger()

def get_database_engine() -> Engine:
    """Reads database credentials from .env file and creates a SQLAlchemy engine."""
    load_dotenv()

    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT")
    database = os.getenv("DB_NAME")
    
    if not all([user, password, host, port, database]):
        logger.error("Database credentials missing in .env file.")
        raise ValueError("Missing DB credentials.")

    logger.info("Creating database engine")
    try:
        connection_string = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
        engine = create_engine(connection_string)
        logger.debug(f"Loaded credentials -> User: {user}, Host: {host}:{port}, DB: {database}")
        logger.info("Database engine created successfully")
        return engine
    except Exception as e:
        logger.error(f"Error occurred while creating database engine: {e}")
        raise



def extract_data(query: str) -> pd.DataFrame:
    """Executes the provided SQL query, extracts data from the database, and saves it to a CSV file.
    
    Args:
        query (str): The SQL query to execute.
        file_name (str): The path to the CSV file where the data will be saved.
    returns:
        None
    """
    logger.info("Extracting data from database")
    try:
        engine = get_database_engine()
        
        logger.debug(f"Executing Query: {query}")
        data = pd.read_sql_query(query, engine)
        logger.debug(f"Extracted Data Shape: {data.shape}")
        logger.info(f"Data extracted  successfully")
        return data
   
    except Exception as e:
        logger.error(f"Error occurred while extracting data to CSV: {e}")
        raise





def split_data(data,test_size:float,random_state:int) -> tuple[pd.DataFrame,pd.DataFrame]:
    """Splits the data into training and testing sets.
    
    Args:
        X (pd.DataFrame): The input features to split.
        y (pd.Series): The target variable to split.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): Controls the randomness of the split.
        """
    
    logger.info("Splitting data into training and testing sets")
    try: 
        unique_customers = data['customer_unique_id'].unique()
        train_ids, test_ids = train_test_split(unique_customers, test_size=test_size, random_state=random_state)
        
        train_df = data[data['customer_unique_id'].isin(train_ids)]
        test_df = data[data['customer_unique_id'].isin(test_ids)]
            
        
        logger.debug(f"Training Data Shape: {train_df.shape}, Testing Data Shape: {test_df.shape}")
        return train_df,test_df
    except Exception as e:
        logger.error(f"Error occurred while splitting data: {e}")
        raise
    



def save_data_to_csv(data:Union[pd.DataFrame, pd.Series], file_name:str,file_path:str) -> None:
    """Saves the provided DataFrame to a CSV file.
    
    Args:
        data (pd.DataFrame): The data to save.
        file_name (str): The path to the CSV file where the data will be saved.
        file_path (str): The directory path where the CSV file will be saved.
    Returns:    
        None
    """
    logger.info("Saving data to CSV")
    try:
        os.makedirs(file_path, exist_ok=True)
        full_path = os.path.join(file_path, f"{file_name}.csv")

        data.to_csv(full_path, index=False)  
        logger.info(f"Data saved to {full_path} successfully")
    except Exception as e:
        logger.error(f"Error occurred while saving data to CSV: {e}")
        raise




def load_data_from_csv(file_name:str) -> pd.DataFrame:
    """Loads data from a CSV file into a DataFrame.
    
    Args:
        file_name (str): The path to the CSV file to load.  
    Returns:
        pd.DataFrame: The loaded data as a DataFrame."""
    logger.info(f"Loading data from {file_name}")
    try:
        data = pd.read_csv(file_name)
        logger.debug(f"Loaded Data Shape: {data.shape}")
        return data
    except Exception as e:
        logger.error(f"Error occurred while loading data from CSV: {e}")
        raise

def main():
    try:
        logger.info("Starting data ingestion process")
        file_path = './data/raw/raw_data.csv'
        data = None
        if not os.path.exists(file_path):
            os.makedirs('./data/raw', exist_ok=True)
            
            query = """
            WITH Orders_detail AS (
            SELECT 
                o.order_id,
                o.order_purchase_timestamp,
                o.order_status,
                c.customer_id,
                c.customer_unique_id,
                c.customer_city,
                c.customer_state,
                p.payment_type,
                p.payment_installments,
                p.payment_value,
				items.price,
				items.freight_value,
				reviews.review_score
            FROM 
                olist_orders_dataset AS o
            INNER JOIN 
                olist_customers_dataset AS c 
                ON o.customer_id = c.customer_id
            INNER JOIN 
                olist_order_payments_dataset AS p 
                ON o.order_id = p.order_id
			
			INNER JOIN
				olist_order_items_dataset as items
				ON
				items.order_id = o.order_id
			INNER JOIN
				olist_order_reviews_dataset as reviews
				ON
				reviews.order_id = o.order_id
            WHERE 
					o.order_status = 'delivered'    
            )

                SELECT
                    customer_unique_id,

                    -- Features (X)
                    EXTRACT(DAY FROM ('2018-10-17'::date - MAX(order_purchase_timestamp))) AS recency,
                    SUM(payment_value) AS total_payment,
                    ROUND(AVG(payment_installments), 1) AS avg_installments,
                    MAX(customer_city) AS customer_city,
                    MAX(customer_state) AS customer_state,
					Count(order_id) as frequency,
					ROUND(Avg(review_score), 1) as avg_review,

                    CASE 
                        WHEN EXTRACT(DAY FROM ('2018-08-01'::date - MAX(order_purchase_timestamp))) > 349 THEN 1
                        ELSE 0 
                    END AS churn_status

                FROM 
                    Orders_detail
                GROUP BY 
                    customer_unique_id;

            """ 
                       
            data = extract_data(query)
            save_data_to_csv(data, 'raw_data', './data/raw')
        else:
            logger.info("Raw data already exists. Skipping extraction.")
        params = load_yaml('./params.yaml')
        test_size = params['data_ingestion']['test_size']
        random_state = params['data_ingestion']['random_state']
        data = load_data_from_csv(file_path)
        X_train, X_test = split_data(data, test_size=test_size, random_state=random_state)
        save_data_to_csv(X_train, 'X_train', './data/raw')
        logger.debug(f"Train data saved with shape: {X_train.shape}")
        save_data_to_csv(X_test, 'X_test', './data/raw')
        logger.debug(f"Test data saved with shape: {X_test.shape}")
    
        logger.info("Data ingestion process completed successfully")
    except Exception as e:
        logger.error(f"Error occurred in main function: {e}")
        raise    
        
if __name__ == "__main__":
    main()