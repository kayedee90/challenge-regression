import pandas as pd
print("LOADED FUNCTION MODULE:", __file__)
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
        "locality"]
        cleaned_columns = df.drop(columns=Col_to_drop, axis=1)
        print(cleaned_columns.info())
        return cleaned_columns
    
    def column_name_change(self, df) :
        df.rename(columns={'MunicipalityCleanName': 'Municipality'},inplace=True)
        return df
    
    def save_csv(self, df, path):
        df.to_csv(path, index=False)