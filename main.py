from functions import DataAnalysis 

def main():
    dataviz = DataAnalysis()

    df = dataviz.read_csv("./raw_data_sm.csv")
    
    df = dataviz.drop_column(df)

    df = dataviz.column_name_change(df)

    dataviz.save_csv(df, 'cleaned_data_for_model.csv')
    
    print(df.info())


if __name__ == "__main__":
    main()