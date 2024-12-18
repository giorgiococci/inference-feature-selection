import argparse
import os
import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline


def main(output_path):
    iris = load_iris()
    pipe = Pipeline(steps=[
    ('select', SelectKBest(k=2)),
    ('clf', LogisticRegression())])
    pipe.fit(iris.data, iris.target)
    pipe[:-1].get_feature_names_out()

    # Get the selected feature names
    selected_features = pipe.named_steps['select'].get_feature_names_out(iris.feature_names)
    print("Selected features:", selected_features)
    
    # Set up MLFlow tracking
    mlflow.set_tracking_uri("file:///C:/repo/San Raffaele/inference-feature-selection/mlruns")
    mlflow.set_experiment("Iris Feature Selection")
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("k", 2)
        mlflow.log_text("selected_features", ",".join(selected_features))
        # Log the model
        mlflow.sklearn.log_model(pipe, "model")
        
        # Save the model to a folder
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        mlflow.sklearn.save_model(pipe, output_path)
        
        print(f"Model saved to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=int, default="C:/repo/San Raffaele/inference-feature-selection/models/iris-model")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args.output_path)