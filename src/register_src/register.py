import argparse
import os
import mlflow
import mlflow.sklearn
from mlflow.client import MlflowClient
from pathlib import Path
# from azureml.core import Workspace, Model
from sklearn.pipeline import Pipeline

def register(model_path: Path, model_name: str, pipeline_steps_to_exclude: str):
    mlflow.start_run()
    # Load the model from the specified folder
    model = mlflow.sklearn.load_model(model_path)
    if pipeline_steps_to_exclude:
        pipeline_steps_to_exclude_list = pipeline_steps_to_exclude.split(",")

        # Remove the feature selection step from the pipeline
        if isinstance(model, Pipeline):
            steps = [(name, step) for name, step in model.steps if name not in pipeline_steps_to_exclude_list]
            model = Pipeline(steps=steps)

    # Register the model in Azure ML
    model_description = "Logistic regression model trained on Iris dataset without feature selection step"
    mlflow.sklearn.log_model(model, artifact_path="output_model")
    
    tags = {
        "task": "regression",
        "framework": "scikit-learn",
        "step": "no-select",
        "description": model_description
    }
    
    run = mlflow.active_run()
    model_uri = f"runs:/{run.info.run_id}/output_model"
    mv = mlflow.register_model(model_uri, model_name, tags=tags)
    print(f"Name: {mv.name}")
    print(f"Version: {mv.version}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=Path)
    parser.add_argument("--model_name", type=str, default="iris_model_no_select")
    parser.add_argument("--pipeline_steps_to_exclude", type=str, default="select", required=False)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    register(args.model_path, args.model_name, args.pipeline_steps_to_exclude)
