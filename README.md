# mlflow_tutorials

Run "mlflow ui" in the parent directory to run the ui locally

In case that i want to use a set of dataset to evaluate the model: 
```python
import mlflow
import mlflow.sklearn
import subprocess
import os
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Set the experiment name
mlflow.set_experiment('Iris_Random_Forest')

# Parameters for the Random Forest Classifier
params = {"classifier__n_estimators": 100, "classifier__max_depth": 2, "classifier__random_state": 42}

# Get code version (git commit hash)
code_version = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")

# Get data version (assuming data is version controlled with DVC)
data_version = subprocess.check_output(["dvc", "version"]).strip().decode("utf-8")

# Directory containing the datasets
data_dir = "data"

for filename in os.listdir(data_dir):
    if filename.endswith(".csv"): # assuming the datasets are csv files
        df = pd.read_csv(os.path.join(data_dir, filename))
        
        X = df.iloc[:, :-1] # assuming the last column is the target
        y = df.iloc[:, -1]

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        with mlflow.start_run():

            # Create a Pipeline
            pipe = Pipeline([
                ('scaler', StandardScaler()), 
                ('classifier', RandomForestClassifier())
            ])

            # Train the pipeline (preprocessing + model)
            pipe.fit(X_train, y_train)

            # Make predictions
            y_pred = pipe.predict(X_test)

            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)

            # Calculate precision, recall, and F1 score
            report = classification_report(y_test, y_pred, output_dict=True)
            precision = report["macro avg"]["precision"]
            recall = report["macro avg"]["recall"]
            f1_score = report["macro avg"]["f1-score"]

            # Log model
            mlflow.sklearn.log_model(pipe, "model")

            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1_score)

            # Log parameters
            mlflow.log_params(params)

            # Log versions
            mlflow.set_tag("code_version", code_version)
            mlflow.set_tag("data_version", data_version)

            # Log model description
            model_description = "A pipeline consisting of a StandardScaler and a RandomForestClassifier."
            mlflow.set_tag("model_description", model_description)

            # Log dataset name
            mlflow.set_tag("dataset", filename)

            print("Model saved in run %s" % mlflow.active_run().info.run_uuid)
```