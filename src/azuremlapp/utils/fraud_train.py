#fraud detection model training
import csv
import os
from dotenv import load_dotenv
import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(env_path)


def _detect_delimiter(file_path):
    with open(file_path, newline='') as file:
        sample = file.read(1024)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=',;\t|')
            return dialect.delimiter
        except csv.Error:
            return ','


def fraud_detection_train(file_path=None):
    if not file_path:
        file_path = os.getenv('file_path')

    if not file_path:
        file_path = os.path.join(os.path.dirname(__file__), '..', 'resources', 'fraud_transaction.csv')

    file_path = os.path.expanduser(file_path)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Training file not found: {file_path}")

    delimiter = _detect_delimiter(file_path)
    df = pd.read_csv(file_path, delimiter=delimiter)
    if len(df.columns) == 1 and '\t' in df.columns[0]:
        df = pd.read_csv(file_path, delimiter='\t')

    expected_columns = {'transaction_id', 'merchant_category', 'location', 'label'}
    missing_columns = expected_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing expected columns {sorted(missing_columns)} in data. "
            f"Found columns: {list(df.columns)}"
        )

    # Preprocess the data (example: drop missing values)
    df.dropna(inplace=True)
    # one hot encoding for categorical variables like merchant_category and location
    df = pd.get_dummies(df, columns=['merchant_category', 'location'], drop_first=True)


    # Split the data into features and target variable
    X = df.drop(['transaction_id','label'], axis=1)  # Assuming 'label' is the target column
    y = df['label']

    #split the data into training and testing sets
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Further steps for model training would go here (e.g., train-test split, model fitting, etc.)
    
    #decision tree classifier
    
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    print(f"Model accuracy: {accuracy:.2f}")
    #calculate f1 score
    
    f1 = f1_score(y_test, y_pred)
    print(f"Model F1 score: {f1:.2f}")

    #calculate precision and recall
    
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    print(f"Model Precision: {precision:.2f}")
    print(f"Model Recall: {recall:.2f}")

    #calculate confusion matrix
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:\n{cm}")

    model_path = os.getenv('model_path', 'outputs/fraud_model.pkl')
    model_path = os.path.expanduser(model_path)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    artifact = {
        "model": clf,
        "feature_columns": X.columns.tolist(),
        "metrics": {
            "accuracy": accuracy,
            "f1_score": f1,
            "precision": precision,
            "recall": recall
        }
    }

    joblib.dump(artifact, model_path)

    print(f"\nModel saved to: {model_path}")

    return model_path



if __name__ == "__main__":
    fraud_detection_train()