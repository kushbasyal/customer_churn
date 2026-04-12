import pandas as pd
import numpy as np

class FeatureEngineer:

    def __init__(self, df):
        self.df = df

    def create_features(self):

        if self.df is not None:

            # 🔥 Drop unnecessary columns
            self.df = self.df.drop(columns=['customer_id', 'total_charges'], errors='ignore')
            return self.df

        else:
            print("No data for feature engineering")
            return None