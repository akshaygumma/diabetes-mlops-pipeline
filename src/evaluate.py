import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import yaml
import os
import mlflow

from urllib.parse import urlparse  ##Used to parse the model registry URI - Get the schema if the remote repository 

os.environ['MLFLOW_TRACKING_URI'] = "https://dagshub.com/akshaygumma/ml-pipeline.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME'] = "akshaygumma"
os.environ['MLFLOW_TRACKING_PASSWORD'] = "ec7d7a0ab9eef7e04dd6e31c5d6c4818e7df2843"


params = yaml.safe_load(open("params.yaml"))["train"]

def evaluate_model(input_path, model_path):
    data = pd.read_csv(input_path)
    X = data.drop(columns=['Outcome'], axis=1)
    y = data['Outcome']

    #Set Tracking URI
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    #Loading the model from pickle file
    model = pickle.load(open(model_path, 'rb'))

    predictions = model.predict(X)
    accuracy = accuracy_score(y,predictions)
    class_report = classification_report(y, predictions)
    conf_matrix = confusion_matrix(y, predictions)

    #Logging the metrics to mlflow
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_text(class_report, "classification_report.txt")
    mlflow.log_text(str(conf_matrix), "confusion_matrix.txt")
    print(f"Accuracy: {accuracy}, Classification Report: {class_report}, Confusion Matrix: {conf_matrix}")

if __name__ == "__main__":
    evaluate_model(params["data"], params["model"])