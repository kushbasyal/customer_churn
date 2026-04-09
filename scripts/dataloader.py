import pandas as pd
from config_paths import RAW_DATA_PATH

def load_data():
    '''Load the raw customer churn dataset'''
    try:
        df = pd.read_csv(RAW_DATA_PATH)
        print("Data Load Successfully")
        return df
    except FileNotFoundError:
        print("File not found :", RAW_DATA_PATH)
        return None
    except Exception as e:
        print("Error loading data", e)
        return None
    
# Call the function and use its return value
df = load_data()

if df is not None:
    print(df.head())
else:
    print("File couldnot be loaded")