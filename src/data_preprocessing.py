import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from imblearn.combine import SMOTEENN

from utils.logger import Logger
from utils.yaml_loader import load_yaml
from src.data_ingestion import save_data_to_csv, load_data_from_csv
from utils.utility_functions import save_object, save_numpy_array_data

logger = Logger('data_preprocessing', level="DEBUG").get_logger()

def data_clean(data: pd.DataFrame) -> pd.DataFrame:
    try:
        logger.debug("Cleaning duplicates.")
        return data.drop_duplicates()
    except Exception as e:
        logger.error(f"Error in data_clean: {e}")
        raise

def drop_unwanted_columns(data: pd.DataFrame) -> pd.DataFrame:
    try:
        # Direct columns drop
        cols_to_drop = ['customer_unique_id', 'recency', 'customer_city', 'customer_state']
        data = data.drop(columns=cols_to_drop, errors='ignore')
        logger.debug(f"Dropped columns: {cols_to_drop}")
        return data
    except Exception as e:
        logger.error(f"Error in drop_unwanted_columns: {e}")
        raise

def get_data_transformer_object() -> ColumnTransformer:
    try:
        logger.info("Initializing Data Transformer Pipeline")
        params_yaml = load_yaml('./params.yaml')
        
        std_features = params_yaml['data_preprocessing']['std_features']
        min_max_features = params_yaml['data_preprocessing']['min_max_features']

        std_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("std_scaler", StandardScaler())
        ])

        minmax_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("minmax_scaler", MinMaxScaler())
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('StandardScaler', std_pipeline, std_features),
                ('MinMaxScaler', minmax_pipeline, min_max_features)
            ],
            remainder="passthrough"
        )
        return preprocessor
    except Exception as e:
        logger.error(f"Error in get_data_transformer_object: {e}")
        raise

def main():
    try:
        uncleaned_data_path = './data/raw'
        processed_data_path = './data/processed'
        TARGET_COLUMN = 'churn_status'

        logger.info("Starting data preprocessing process")
        
        # 1. Load Data
        train_df = load_data_from_csv(os.path.join(uncleaned_data_path, 'X_train.csv'))
        test_df = load_data_from_csv(os.path.join(uncleaned_data_path, 'X_test.csv'))

        # 2. Clean & Drop Columns
        train_df = data_clean(train_df)
        test_df = data_clean(test_df)

        X_train = drop_unwanted_columns(train_df.drop(columns=[TARGET_COLUMN]))
        y_train = train_df[TARGET_COLUMN]
        
        X_test = drop_unwanted_columns(test_df.drop(columns=[TARGET_COLUMN]))
        y_test = test_df[TARGET_COLUMN]

        # 3. Apply Scaling
        preprocessor = get_data_transformer_object()
        
        logger.info("Applying fit_transform on Train and transform on Test")
        train_arr_preprocessed = preprocessor.fit_transform(X_train)
        test_arr_preprocessed = preprocessor.transform(X_test)

        # 4. Handle Imbalance (Only on Train)
        logger.info("Applying SMOTEENN on Training data")
        smt = SMOTEENN(sampling_strategy="minority", random_state=42)
        X_train_final, y_train_final = smt.fit_resample(train_arr_preprocessed, y_train)

        # 5. Merge Features and Target
        # FIX: Test data ko bhi array format mein convert karna hoga after transformation
        train_final_arr = np.c_[X_train_final, np.array(y_train_final)]
        test_final_arr = np.c_[test_arr_preprocessed, np.array(y_test)]

        # 6. Save Artifacts
        logger.info("Saving Preprocessor and Numpy arrays")
        os.makedirs('./artifacts', exist_ok=True)
        os.makedirs(processed_data_path, exist_ok=True)

        save_object(file_path='./artifacts/preprocessor/preprocessor.pkl', obj=preprocessor)
        
        # Save as .npy files
        np.save(os.path.join(processed_data_path, 'train.npy'), train_final_arr)
        np.save(os.path.join(processed_data_path, 'test.npy'), test_final_arr)

        logger.info("✅ Data preprocessing completed successfully!")

    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        raise

if __name__ == "__main__":
    main()