
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import matplotlib.pyplot as plt

class ChurnModel:
    
    def __init__(self, df):
        self.df = df
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.preprocessor = None
        self.models = {}
        self.pipelines = {}
        self.results = {}
        self.label_encoder = LabelEncoder()

    
    def prepare_data(self):
        self.X = self.df.drop('churn', axis=1)
        self.y = self.df['churn']

        self.num_cols = self.X.select_dtypes(include='number').columns
        self.cat_cols = self.X.select_dtypes(include='object').columns

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=0
        )

        self.y_train = self.label_encoder.fit_transform(self.y_train)
        self.y_test = self.label_encoder.transform(self.y_test)

        print("Data prepared")

    def build_preprocessor(self):
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.num_cols),
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), self.cat_cols)
            ]
        )
        print("Preprocessor ready")

    def load_models(self):
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=0),
            'KNN': KNeighborsClassifier(),
            'SVM Linear': SVC(kernel='linear', random_state=0),
            'SVM RBF': SVC(kernel='rbf', random_state=0),
            'Naive Bayes': GaussianNB(),
            'Decision Tree': DecisionTreeClassifier(random_state=0),
            'Random Forest': RandomForestClassifier(random_state=0),
            'XGBoost': xgb.XGBClassifier(random_state=0)
        }
        print("Models loaded")

    def train_models(self):
        for name, model in self.models.items():
            pipeline = Pipeline([
                ('preprocessor', self.preprocessor),
                ('classifier', model)
            ])

            pipeline.fit(self.X_train, self.y_train)
            self.pipelines[name] = pipeline

            y_train_pred = pipeline.predict(self.X_train)
            y_test_pred = pipeline.predict(self.X_test)

            train_acc = accuracy_score(self.y_train, y_train_pred)
            test_acc = accuracy_score(self.y_test, y_test_pred)

            self.results[name] = {
                'Train Accuracy': train_acc,
                'Test Accuracy': test_acc
            }

            print(f"{name}: Train={train_acc:.4f}, Test={test_acc:.4f}")

    def evaluate_xgboost(self):
        xgb_pipeline = self.pipelines['XGBoost']

        y_pred = xgb_pipeline.predict(self.X_test)

        print("\nXGBoost Report:")
        print(classification_report(self.y_test, y_pred))

        ConfusionMatrixDisplay.from_estimator(xgb_pipeline, self.X_test, self.y_test)
        plt.show()

        cv_scores = cross_val_score(xgb_pipeline, self.X_train, self.y_train, cv=5)

        print(f"CV Mean: {cv_scores.mean():.4f}")
        print(f"CV Std: {cv_scores.std():.4f}")

    def predict_risk(self, customer_dict):
        df = pd.DataFrame([customer_dict])

        model = self.pipelines['XGBoost']

        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0][1]

        result = "High Risk" if pred == 1 else "Not High Risk"

        return f"{result} (Probability: {prob:.2%})"