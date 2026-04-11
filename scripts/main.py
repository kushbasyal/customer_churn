from dataloader import DataLoader
from data_cleaner import DataCleaner
from feature_engineering import FeatureEngineer
from config_paths import RAW_DATA_PATH

def main():

    # 1. Load data
    loader = DataLoader(RAW_DATA_PATH)
    df = loader.load_data()

    if df is not None:

        print("\n🔍 Raw Data:")
        print(df.head())

        # 2. Inspect
        cleaner = DataCleaner(df)
        cleaner.inspect_data()

        # 3. Clean
        df = cleaner.clean_data()

        print("\n Cleaned Data:")
        print(df.head())

        # 4. Feature Engineering
        fe = FeatureEngineer(df)
        df = fe.create_features()

        print("\n Final Data with Feature:")
        print(df.head())

    else:
        print(" Data not loaded")

if __name__ == "__main__":
    main()