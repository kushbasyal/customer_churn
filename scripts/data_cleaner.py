class DataCleaner:

    def __init__(self, df):
        self.df = df

    # =========================
    # 1. INSPECT DATA
    # =========================
    def inspect_data(self):
        if self.df is not None:

            print(f"\n Shape: {self.df.shape}")
            print(f"🔁 Duplicates: {self.df.duplicated().sum()}")

            print("\n Missing Values:")
            print(self.df.isna().sum())

            print(f"\n Total Missing: {self.df.isna().sum().sum()}")

            print("\n Data Types:")
            print(self.df.dtypes)

            print("\n🔍 First 5 rows:")
            print(self.df.head())

            return self.df

        else:
            print(" No data to inspect")
            return None

    # =========================
    # 2. CLEAN DATA
    # =========================
    def clean_data(self):
        if self.df is not None:

            # 1. Strip column names
            self.df.columns = self.df.columns.str.strip()

            # 2. Remove duplicates
            self.df = self.df.drop_duplicates()

            # 3. Separate columns
            numerical_columns = self.df.select_dtypes(include='number').columns
            categorical_columns = self.df.select_dtypes(include='object').columns

            # 4. Fill numeric missing → mean
            for col in numerical_columns:
                self.df[col] = self.df[col].fillna(self.df[col].mean())

            # 5. Fill categorical missing → "Unknown"
            for col in categorical_columns:
                if len(self.df[col].mode()) > 0:
                    self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
                else:
                    self.df[col] = self.df[col].fillna("Unknown")

            print(" Data cleaned successfully")

            return self.df

        else:
            print(" No data to clean")
            return None