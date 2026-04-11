import pandas as pd
import numpy as np

class DataLoader:

    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    def load_data(self):
        try:
            self.df = pd.read_csv(self.file_path)
            print("✅ Data loaded successfully")
            return self.df

        except FileNotFoundError:
            print("❌ File not found:", self.file_path)
            return None

        except Exception as e:
            print("❌ Error loading data:", e)
            return None