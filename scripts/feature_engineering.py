import pandas as pd
import numpy as np

from data_cleaner import load_data, clean_data, inspect_data

def create_features(df):
    if df is not None:
        # Feature 1: Average monthly spend
        df['avg_monthly_spend'] = df['monthly_charges']  # Default to monthly_charges
        df.loc[df['tenure'] > 0, 'avg_monthly_spend'] = df['total_charges'] / df['tenure']
        
        # Feature 2: High risk customer flag
        df['high_risk_customer'] = (
            (df['contract'] == 'Month-to-month') & 
            (df['tech_support'] == 'No') & 
            (df['support_calls'] >= 2)
        ).astype(int)
        
        print("Feature engineering completed")
        return df
    return None


# Review the details summary
df = load_data()

if df is not None:
    df = clean_data(df)
    df = create_features(df)
    print(df.head())
    print(df[['avg_monthly_spend', 'high_risk_customer']].head(10))
else:
    print("No data to process")