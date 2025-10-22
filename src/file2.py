import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import dagshub

# Initialize DagsHub connection
dagshub.init(repo_owner='Vish043', repo_name='MLOPS_Experiments_with_MLFLOW', mlflow=True)

# Use DagsHub MLflow tracking
mlflow.set_tracking_uri("https://dagshub.com/Vish043/MLOPS_Experiments_with_MLFLOW.mlflow")

wine = load_wine()
x = wine.data
y = wine.target

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=42)

max_depth = 8
n_estimators = 5

mlflow.set_experiment('MLOPS_Experiment-2')

with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('n_estimators', n_estimators)

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig("Confusion-matrix.png")

    # Log artifacts
    mlflow.log_artifact("Confusion-matrix.png")
    mlflow.log_artifact(__file__)

    # Tags
    mlflow.set_tags({"Author": "Vikash", "Project": "Wine Classification"})

    # ✅ Save and log model manually (this works on DagsHub)
    joblib.dump(rf, "RandomForestModel.pkl")
    mlflow.log_artifact("RandomForestModel.pkl")

    print("Accuracy:", accuracy)
