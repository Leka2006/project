# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# 1. Load the data
def load_data(filepath):
    df = pd.read_csv(filepath)
    print(f"Data Loaded. Shape: {df.shape}")
    return df

# 2. Preprocess the data
def preprocess_data(df):
    # Example: Fill missing values
    df = df.fillna(method='ffill')

    # Encode categorical variables
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])

    return df

# 3. Explore hidden patterns
def explore_patterns(df, target_column):
    plt.figure(figsize=(12, 8))
    corr = df.corr()
    sns.heatmap(corr, cmap='coolwarm', annot=False)
    plt.title('Feature Correlation Heatmap')
    plt.show()

    # Feature Importance later after model training

# 4. Train model
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# 5. Evaluate model
def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, preds))

    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

# 6. Feature Importance
def feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(12, 6))
    plt.title("Feature Importance")
    sns.barplot(x=[feature_names[i] for i in indices], y=importances[indices])
    plt.xticks(rotation=90)
    plt.show()

# 7. Main Execution
def main(filepath, target_column):
    df = load_data(filepath)
    df = preprocess_data(df)

    explore_patterns(df, target_column)

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)

    feature_importance(model, X.columns)

# Run the program
if __name__ == "__main__":
    # Example: Replace 'your_data.csv' and 'Churn' with your actual file and column name
    main(filepath='your_data.csv', target_column='Churn')