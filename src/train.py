import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from mlflow.models import infer_signature
import os
import mlflow
from urllib.parse import urlparse  ##Used to parse the model registry URI - Get the schema if the remote repository 

os.environ['MLFLOW_TRACKING_URI'] = "https://dagshub.com/akshaygumma/ml-pipeline.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME'] = "akshaygumma"
os.environ['MLFLOW_TRACKING_PASSWORD'] = "ec7d7a0ab9eef7e04dd6e31c5d6c4818e7df2843"


def hyperparameter_tuning(X_train, y_train, param_grid):
    rf = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search

#Loading the params from the params.yaml file
params = yaml.safe_load(open("params.yaml"))["train"]

def train_model(input_path,model_path, random_state, n_estimators, max_depth):
    data = pd.read_csv(input_path)
    X = data.drop(columns=['Outcome'], axis=1)
    y = data['Outcome']

    #Set Tracking URI
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    ##Start an mlflow run
    with mlflow.start_run():
        #split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

        signature = infer_signature(X_train, y_train)

        #Define the hyperparameter grid
        param_grid = {
            'n_estimators': [100,200],
            'max_depth': [5,10,None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }


        #Perform hyperparameter tuning
        grid_search = hyperparameter_tuning(X_train, y_train, param_grid)
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        print(f"Best Hyperparameters: {best_params} , Best Model: {best_model}")


        #Predict and evaluate the model
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        print(f"Accuracy: {accuracy}")
        print(f"Classification Report:\n{class_report}")
        print(f"Confusion Matrix:\n{conf_matrix}")


        #Log additional metrics
        mlflow.log_metric("accuracy", accuracy)

        mlflow.log_param("best_n_estimators", best_params['n_estimators'])
        mlflow.log_param("best_max_depth", best_params['max_depth'])
        mlflow.log_param("best_min_samples_split", best_params['min_samples_split'])
        mlflow.log_param("best_min_samples_leaf", best_params['min_samples_leaf'])
        
        #Log the confusion matrix as an artifact and classification report as text
        mlflow.log_text(str(conf_matrix),"confusion_matrix.txt")
        mlflow.log_text(class_report,"classification_report.txt")

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        if(tracking_url_type_store != "file"):
            mlflow.sklearn.log_model(best_model, "model", signature=signature, registered_model_name="RandomForestClassifier_BestModel")
        else:
            mlflow.sklearn.log_model(best_model, "model", signature=signature)


        #Create directory if it doesn't exist to save the model locally
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        filename = model_path

        pickle.dump(best_model, open(filename, 'wb'))
        print(f"Model saved to {model_path}")


if __name__ == "__main__":
    train_model(params["data"],params["model"], params["random_state"], params["n_estimators"], params.get("max_depth", None))