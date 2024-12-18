import mlflow.sklearn
import numpy as np

# Load the model from the specified folder
model_path = "C:/repo/San Raffaele/inference-feature-selection/models/iris-model"
model_inference_path = "C:/repo/San Raffaele/inference-feature-selection/models/iris-model-inference"
model = mlflow.sklearn.load_model(model_path)
model_inference = mlflow.sklearn.load_model(model_inference_path)

# Sample dataset (replace with your actual data)
sample_data = np.array([[5.1, 3.5, 3.2, 1.2],
                        [6.2, 3.4, 5.4, 2.3]])

# Predict results using the loaded model
predictions = model.predict(sample_data)
print("Predictions from original model:", predictions)
try:
    predictions_inference = model_inference.predict(sample_data)
except ValueError as ex:
    print("As expected, the inference model failed with the following error:", ex)
    print("Retrying with the correct number of features...")

sample_data_inference = np.array([[5.1, 3.5],
                        [6.2, 3.4]])

predictions_inference = model_inference.predict(sample_data_inference)

print("Predictions from inference model:", predictions_inference)