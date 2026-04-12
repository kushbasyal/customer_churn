from dataloader import DataLoader
from data_cleaner import DataCleaner
from feature_engineering import FeatureEngineer
from churn_model import ChurnModel
from config_paths import RAW_DATA_PATH


def main():

    # 1. Load raw data
    loader = DataLoader(RAW_DATA_PATH)
    df = loader.load_data()

    if df is not None:

        print("\n🔍 Raw Data:")
        print(df.head())

        # 2. Clean data
        cleaner = DataCleaner(df)
        cleaner.inspect_data()
        clean_df = cleaner.clean_data()

        print("\n✅ Cleaned Data:")
        print(clean_df.head())

        # 3. Feature Engineering (drop columns etc.)
        fe = FeatureEngineer(clean_df)
        model_df = fe.create_features()

        print("\n🚀 Model Data (after feature engineering):")
        print(model_df.head())

        # 4. Model Training (USE model_df ONLY)
        model = ChurnModel(model_df)

        model.prepare_data()
        model.build_preprocessor()
        model.load_models()
        model.train_models()
        model.evaluate_xgboost()

        # 5. Prediction Example
        result = model.predict_risk({
            'tenure': 52,
            'monthly_charges': 54.20,
            'contract': 'Month-to-month',
            'payment_method': 'Credit',
            'internet_service': 'DSL',
            'tech_support': 'No',
            'online_security': 'Yes',
            'support_calls': 1
        })

        print("\n🔮 Prediction Result:")
        print(result)

    else:
        print("❌ Data not loaded")


if __name__ == "__main__":
    main()