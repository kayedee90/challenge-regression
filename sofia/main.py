from sofia.functions import DataAnalysis
import os

def main():
    dataviz = DataAnalysis()

    df = dataviz.read_csv("data/raw_data.csv")
    
    df = dataviz.drop_column(df)

    df = dataviz.column_name_change(df)

    dataviz.save_csv(df, 'cleaned_data_for_model.csv')
    
    print(df.info())


if __name__ == "__main__":
    main()