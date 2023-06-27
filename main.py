import mlflow
import mlflow.sklearn
import subprocess
import optuna
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Load iris data
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set the experiment name
mlflow.set_experiment('Iris_Random_Forest_optuna')

# Get code version (git commit hash)
code_version = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")

# Get data version (assuming data is version controlled with DVC)
# data_version = subprocess.check_output(["dvc", "version"]).strip().decode("utf-8")


def objective(trial):
    with mlflow.start_run():
        # Specify the hyperparameter to be optimized
        params = {
            "classifier__n_estimators": trial.suggest_int("n_estimators", 50, 150),
            "classifier__max_depth": trial.suggest_int("max_depth", 2, 10),
        }

        # Create a Pipeline
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(random_state=42))
        ])

        # Train the pipeline (preprocessing + model)
        pipe.set_params(**params)
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
        # mlflow.set_tag("data_version", data_version)

        # Log model description
        model_description = "A pipeline consisting of a StandardScaler and a RandomForestClassifier with hyperparameters optimized by Optuna."
        mlflow.set_tag("model_description", model_description)
        mlflow.set_tag("pipeline", str(pipe.steps))

        return accuracy


# Create a study and run the optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=5)



