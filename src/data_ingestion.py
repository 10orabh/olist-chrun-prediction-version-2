import pandas as pd 
from utils.logger import Logger
from sqlalchemy import create_engine
from dotenv import load_dotenv
from sqlalchemy.engine import Engine
import os

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

def extract_data_to_csv(query: str, file_name: str) -> None:
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
        
        data.to_csv(file_name, index=False)
        logger.info(f"Data extracted and saved to {file_name} successfully")
    except Exception as e:
        logger.error(f"Error occurred while extracting data to CSV: {e}")
        raise

def main():
    try:
        logger.info("Starting data ingestion process")
        file_path = './data/raw/raw_data.csv'
        
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
                    p.payment_value
                FROM 
                    olist_orders_dataset AS o
                INNER JOIN 
                    olist_customers_dataset AS c 
                    ON o.customer_id = c.customer_id
                INNER JOIN 
                    olist_order_payments_dataset AS p 
                    ON o.order_id = p.order_id
                WHERE 
                    o.order_purchase_timestamp < '2018-08-01'::date
                    AND o.order_status = 'delivered'    
            )

                SELECT
                    customer_unique_id,

                    -- Features (X)
                    EXTRACT(DAY FROM ('2018-08-01'::date - MAX(order_purchase_timestamp))) AS recency,
                    SUM(payment_value) AS total_payment,
                    ROUND(AVG(payment_installments), 1) AS avg_installments,
                    MAX(customer_city) AS customer_city,
                    MAX(customer_state) AS customer_state,

                    -- Target Option 1
                    CASE 
                        WHEN COUNT(DISTINCT order_id) > 1 THEN 1 
                        ELSE 0 
                    END AS is_repeat_purchase, 

                    -- Target Option 2
                    CASE 
                        WHEN EXTRACT(DAY FROM ('2018-08-01'::date - MAX(order_purchase_timestamp))) > 120 THEN 1
                        ELSE 0 
                    END AS churn_status

                FROM 
                    Orders_detail
                GROUP BY 
                    customer_unique_id;

            """ 
            extract_data_to_csv(query, file_path)
        else:
            logger.info("Raw data already exists. Skipping extraction.")

    except Exception as e:
        logger.error(f"Error occurred in main function: {e}")
        raise    
        
if __name__ == "__main__":
    main()