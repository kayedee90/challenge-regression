from functions import DataAnalysis
import pandas as pd
from pathlib import Path

def main():
    dataviz = DataAnalysis()

    # Read raw data
    raw_path = Path(__file__).resolve().parent.parent / 'data' / 'raw' / 'raw_data_sm.csv'
    df = dataviz.read_csv(raw_path)

    df = dataviz.drop_column(df)
    df = dataviz.building_condition(df)
    df = dataviz.epc_score(df)
    df = dataviz.convert_postcode_tostring(df)

    # Save cleaned data
    processed_path = Path(__file__).resolve().parent.parent / 'data' / 'processed'
    processed_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path / 'cleaned_data.csv', index=False)
    print("✅ Cleaned data saved.")

    # Split train/test and save them
    dataviz.split_train_test(df)

if __name__ == "__main__":
    main()