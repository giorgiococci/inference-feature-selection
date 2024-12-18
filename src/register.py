import argparse
import os
import mlflow
import mlflow.sklearn
from pathlib import Path
# from azureml.core import Workspace, Model
from sklearn.pipeline import Pipeline

def register(model_path: Path, model_name: str, pipeline_steps_to_exclude: str):
    # Load the model from the specified folder
    model = mlflow.sklearn.load_model(model_path)
    pipeline_steps_to_exclude_list = pipeline_steps_to_exclude.split(",")

    # Remove the feature selection step from the pipeline
    if isinstance(model, Pipeline):
        steps = [(name, step) for name, step in model.steps if name not in pipeline_steps_to_exclude_list]
        model = Pipeline(steps=steps)

    # Set up Azure ML workspace
    # ws = Workspace.from_config()

    # Register the model in Azure ML
    model_description = "Logistic regression model trained on Iris dataset without feature selection step"
    model_path = "C:/repo/San Raffaele/inference-feature-selection/models/iris-model-inference"

    # Save the modified model
    mlflow.sklearn.save_model(model, model_path)

    # # Register the model
    # registered_model = Model.register(workspace=ws,
    #                                 model_path=model_path,
    #                                 model_name=model_name,
    #                                 description=model_description)

    # print(f"Model registered: {registered_model.name}, version: {registered_model.version}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="C:/repo/San Raffaele/inference-feature-selection/models/iris-model")
    parser.add_argument("--model_name", type=str, default="iris_model_no_select")
    parser.add_argument("--pipeline_steps_to_exclude", type=str, default="select")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    register(args.model_path, args.model_name, args.pipeline_steps_to_exclude)
