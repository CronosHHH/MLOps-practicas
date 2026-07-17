
# ESTA ES LA OPCION QUE VAMOS A USAR
# NO VAMOS A USAR DECORATOR
# He creado un pipeline mas completo (pipeline.yml)


from azure.ai.ml import MLClient, load_component, load_job
from azure.identity import DefaultAzureCredential
import os

# Autenticación
ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id=os.getenv("AZURE_SUBSCRIPTION_ID"),
    resource_group_name=os.getenv("AZURE_RESOURCE_GROUP"),
    workspace_name=os.getenv("AZURE_WORKSPACE_NAME")
)

loaded_component_prep = load_component(source="./prep.yml")

prep_component = ml_client.components.create_or_update(loaded_component_prep)

pipeline_job = load_job(source="./pipeline.yml")


pipeline_enviado = ml_client.jobs.create_or_update(
    pipeline_job,
    experiment_name="limpieza_datos_personalizado"
)
