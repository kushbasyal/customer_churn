import pandas as pd
import numpy as np
class FeatureEngineer:

    def __init__(self, df):
        self.df = df

    def create_features(self):

        if self.df is not None:

            # avg monthly spend (simple pandas way)
            self.df['avg_monthly_spend'] = self.df['total_charges'] / self.df['tenure']

            # replace invalid values (inf, NaN)
            self.df['avg_monthly_spend'] = self.df['avg_monthly_spend'].fillna(0)

            print(" Feature created: avg_monthly_spend")

            return self.df

        else:
            print(" No data for feature engineering")
            return None