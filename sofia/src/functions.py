import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import make_column_selector as selector
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from pathlib import Path
import numpy as np
import pickle


class DataAnalysis:
        
    def read_csv(self, path):
        # read the csv file with pandas
        df = pd.read_csv(path)
        print(df.info())
        return df

    def drop_column(self, df) :
        # dropping columns that are not needed 
        Col_to_drop = ["id",
        "url",
        "locality", 'MunicipalityCleanName','price_square_meter']
        df = df.drop(columns=Col_to_drop, axis=1)
        print(df.info())
        return df
    
    def building_condition(self, df) :
        condition_mapping = {
    'AS_NEW': 5,
    'JUST_RENOVATED': 4,
    'GOOD': 3,
    'TO_RENOVATE': 2,
    'TO_BE_DONE_UP': 2,
    'TO_RESTORE': 1
    }
        df['buildingCondition'] = df['buildingCondition'].replace(condition_mapping)
        return df

    def epc_score(self, df) :
        epcscore_mapping = {
    'A++': 9,
    'A+': 8,
    'A': 7,
    'B': 6,
    'C': 5,
    'D': 4,
    'E': 3,
    'F': 2,
    'G': 1,
    }
        df['epcScore'] = df['epcScore'].replace(epcscore_mapping)
        return df

    def convert_postcode_tostring (self,df):
        df['postCode'] = df['postCode'].astype(str)
        print(df.info())
        return df

    def identify_column_types(self, df):
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        print(f"✅ Numeric columns: {numeric_cols}")
        print(f"✅ Categorical columns: {categorical_cols}")
        return numeric_cols, categorical_cols

    def save_csv(self, df, path):
        df.to_csv(path, index=False)

    
    # Function to split the dataset in train and test
    def split_train_test(self, df, test_size=0.2, random_state=42):
        
        train_set, test_set = train_test_split(df, test_size=test_size, random_state=random_state)

        # Path setup
        base_path = Path(__file__).resolve().parent.parent
        processed_path = base_path / 'data' / 'processed'
        processed_path.mkdir(parents=True, exist_ok=True)  

        # Save the splits explicitly to the processed folder
        train_set_path = processed_path / "train_set.csv"
        test_set_path = processed_path / "test_set.csv"

        train_set.to_csv(train_set_path, index=False)
        test_set.to_csv(test_set_path, index=False)

        print(f"✅ Data split completed: {len(train_set)} training rows, {len(test_set)} test rows.")
        print(f"Train set saved to: {train_set_path}")
        print(f"Test set saved to: {test_set_path}")

        return train_set, test_set