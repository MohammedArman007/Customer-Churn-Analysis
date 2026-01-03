# utils.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from typing import Tuple, List

def load_data(path: str) -> pd.DataFrame:
    """Load CSV into DataFrame."""
    df = pd.read_csv(path)
    return df

def summary_stats(df: pd.DataFrame) -> dict:
    """Return basic KPIs for the dashboard."""
    total_customers = len(df)
    churn_rate = df['Churn'].mean()
    avg_monthly = df['MonthlyCharges'].mean()
    avg_tenure = df['Tenure'].mean()
    return {
        'total_customers': int(total_customers),
        'churn_rate': float(churn_rate),
        'avg_monthly': float(avg_monthly),
        'avg_tenure': float(avg_tenure)
    }

def build_preprocessor(numeric_features: List[str], categorical_features: List[str]):
    """Return sklearn ColumnTransformer for preprocessing."""
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )
    return preprocessor

def train_save_model(df: pd.DataFrame, model_path: str = 'model.joblib', pipeline_path: str = 'pipeline.joblib'):
    """Train a RandomForest model with a preprocessing pipeline and save both."""
    # Define features and target
    target = 'Churn'
    drop_cols = ['CustomerID'] if 'CustomerID' in df.columns else []
    X = df.drop(columns=[target] + drop_cols)
    y = df[target]

    # Specify numeric & categorical features for the simple schema
    numeric_features = ['Age', 'Tenure', 'MonthlyCharges', 'TotalSpend', 'NumOfPurchases', 'EngagementScore', 'CustomerSatisfaction', 'SupportTickets']
    categorical_features = [c for c in X.columns if c not in numeric_features]

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    # Model pipeline
    clf = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Fit
    pipeline.fit(X_train, y_train)

    # Save
    joblib.dump(pipeline, model_path)
    # Save preprocessor separately if needed
    joblib.dump(preprocessor, pipeline_path)

    # Return basic metrics
    train_score = pipeline.score(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)
    return {'train_score': train_score, 'test_score': test_score}

def predict_single(pipeline, input_df: pd.DataFrame) -> np.ndarray:
    """Return prediction probabilities and classes for the given rows."""
    probs = pipeline.predict_proba(input_df)[:, 1]
    preds = pipeline.predict(input_df)
    return preds, probs
