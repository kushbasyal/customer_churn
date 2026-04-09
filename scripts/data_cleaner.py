
from dataloader import load_data

def inspect_data(df):
    '''Inspect dataset for duplicates, missing values, and shape'''
    if df is not None:
        print(f"Shape: {df.shape}")
        print(f"Duplicates: {df.duplicated().sum()}")
        print(f"Missing Values:\n{df.isna().sum()}")
        print(f"Total Missing: {df.isna().sum().sum()}")
        print(f"First 5 rows:\n{df.head()}")
        return df
    else:
        print("No dataframe to inspect")
        return None
    
def clean_data(df):
    '''
    clean the customer churn dataset:
    strip column names
    removed duplicates
    handle missing values
    '''
    if df is not None:
        # Strip spaces from column names
        df.columns = df.columns.str.strip()

        # Remove duplicates
        df = df.drop_duplicates()

        # Separate numerical and categorical columns
        numerical_columns = df.select_dtypes(include = 'number').columns
        categorical_columns = df.select_dtypes(include = 'object').columns

        # fill the missing values in numerical columns with mean
        for col in numerical_columns:
            df[col] = df[col].fillna(df[col].mean())

        # fill the missing values in categorical columns with mode(or 'Unknown)
        for col in categorical_columns:
            if len(df[col].mode()) > 0:  # Check if mode exists
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna('Unknown')
        print("Data Cleaned sucessfully")
        return df
    else:
        print("No dataframe to clean")
        return None
    
# Load data and visuals
df = load_data()
if df is not None:
    print("\n🔍 BEFORE CLEANING:")
    inspect_data(df)
    
    df = clean_data(df)
    
    print("\n✨ AFTER CLEANING:")
    inspect_data(df)
else:
    print("Failed to load data")