# Inference Feature Selection

## Run AML pipeline

To run the AML pipeline, use the following script:

```bash
az ml job create --file pipelines/register-model-using-registry.yml --set "inputs.model_input.path=azureml://datastores/workspaceblobstore/paths/azureml/<job-name>/output_model"
```

Here an example:

```bash
az ml job create --file pipelines/register-model-using-registry.yml --set "inputs.model_input.path=azureml://datastores/workspaceblobstore/paths/azureml/e3eaebdc-a8c4-48c1-b1e8-2d01d0c3a628/output_model"
```
