# Componentes 
En vez de tener scripts dispersos, un componente empaqueta ese código para cumplir una función específica.

Usos:
- scripts reutilizables
- compilar pipelines

Se usan para Preparar un codigo Para ser ESCALABLE (mover el modelo a produccion)

### Ejemplos de componentes:

- data prep
  - ingesta de datos (descargar dataset y registrarlo)
  - limpieza de datos (eliminar valores nulos y corregir duplicados)
  - normalización (escalar y transformar los datos)
  - split data (dataset de entrenamiento y dataset de prueba)

- model training
  - entrenar un modelo (Scikit-Learn, TensorFlow, PyTorch, XGBoost, etc.)
  - ajuste de hiperparámetros (hyperparameter tuning)

- evaluation
    - Score Model - acierto modelo en dataset de prueba
    - Ebaluate Model - metricas (accuracy, precision, recall, f1-score, etc.)

- registro/despliegue del modelo
    - poner el modelo en un registro de modelos (Azure ML Model Registry), listo para producción.

### Partes de componentes
* tiene 3 partes:
- 1. Metadatos: nombre, version, descripcion, autor, etc.
- 2. Interfaz: inputs y outputs
    - inputs: datos e hiperparametros
    - outputs: métricas y artefactos   
- 3. Comando, código y entorno: el script que se ejecuta, los parámetros que recibe y el entorno de ejecución (dependencias, librerías, etc.)

### archivos de UN componente
- archivo YAML : archivo de metadatos, interfaz y el comando/codigo/entorno
- script : contiene el flujo de trabajo

El archivo YAML hay 2 formas de crearlo:
- 1. **Enfoque manual**
   - archivo externo, con extensión .yml o .yaml
   - al lado su script de python, con extensión .py
   - hay una separación total entre la infraestructura (el YAML) y el código de Python (el script).

- 2. **Enfoque "Código Primero"**
    - Usar el decorador @command_component()

    - Si no quieres lidiar con la sintaxis de archivos YAML o prefieres no salir de tu entorno de Python (o de tu Jupyter Notebook), el SDK de Azure ML te ofrece un "atajo" usando un **decorador de Python**.


--- 

[EJEMPLO de TODO](componentes_ej/run_pipeline.ipynb)

---

Una canalizacion (pipeline) = flujo de trabajo (workflow) que conecta varios componentes
Estos componentes se pueden conectar de forma secuencial o paralela

Para ejecutar un pipeline
- se ejecutara un JOB de pipeline (pipeline job)
    - en este pipeline job se ejecutaran varios jobs de componentes (component jobs o jobs secundarios)

pipeline 
- se define en un archivo YAML
- incluye:
    - el nombre del trabajo de canalización
    - las entradas
    - las salidas 
    - la configuración

EJEMPLO: 

canalización que 
- prepare primero los datos
- entrene el modelo
```python
from azure.ai.ml.dsl import pipeline

@pipeline()
def pipeline_function_name(pipeline_job_input):
    prep_data = loaded_component_prep(input_data=pipeline_job_input)
    train_model = loaded_component_train(training_data=prep_data.outputs.output_data)

    return {
        "pipeline_job_transformed_data": prep_data.outputs.output_data,
        "pipeline_job_trained_model": train_model.outputs.model_output,
    }
```

Ejemplo de uso de la pipeline de antes:
```python
from azure.ai.ml import Input
from azure.ai.ml.constants import AssetTypes

pipeline_job = pipeline_function_name(
    Input(type=AssetTypes.URI_FILE, 
    path="azureml:data:1"
))
```
Si haces: print(pipeline_job)
saldrá: 
```yml
display_name: pipeline_function_name
type: pipeline
inputs:
  pipeline_job_input:
    type: uri_file
    path: azureml:data:1
outputs:
  pipeline_job_transformed_data: null
  pipeline_job_trained_model: null
jobs:
  prep_data:
    type: command
    inputs:
      input_data:
        path: ${{parent.inputs.pipeline_job_input}}
    outputs:
      output_data: ${{parent.outputs.pipeline_job_transformed_data}}
  train_model:
    type: command
    inputs:
      input_data:
        path: ${{parent.outputs.pipeline_job_transformed_data}}
    outputs:
      output_model: ${{parent.outputs.pipeline_job_trained_model}}
tags: {}
properties: {}
settings: {}
```
--- 
para **EDITAR** nuestro pipeline 
(trás haber usado el decorador @pipeline)

```python
# change the output mode
pipeline_job.outputs.pipeline_job_transformed_data.mode = "upload"
pipeline_job.outputs.pipeline_job_trained_model.mode = "upload"

# set pipeline level compute
pipeline_job.settings.default_compute = "aml-cluster"

# set pipeline level datastore
pipeline_job.settings.default_datastore = "workspaceblobstore"

# set pipeline level datastore
pipeline_job.settings.default_datastore = "workspaceblobstore"
```
Revisar con: 
print(pipeline_job)

enviar trabajo con:
pipeline_job = ml_client.jobs.create_or_update(
    pipeline_job, experiment_name="pipeline_job"
)

---

### Programación de un trabajo de canalización

JobSchedule permite programar la ejecución de un trabajo de canalización en intervalos regulares o en momentos específicos. 
- Programacion basada en tiempo -> RecurrenceTrigger
    - frequency: "Day" | "Week" | "Month"
    - interval: 1 | 2 | 3 | ...

```python
from azure.ai.ml.entities import RecurrenceTrigger

schedule_name = "run_every_minute"

recurrence_trigger = RecurrenceTrigger(
    frequency="minute",
    interval=1,
)
```
pipeline_job para representar la canalización que ha compilado
```python
from azure.ai.ml.entities import JobSchedule

job_schedule = JobSchedule(
    name=schedule_name, trigger=recurrence_trigger, create_job=pipeline_job
)

job_schedule = ml_client.schedules.begin_create_or_update(
    schedule=job_schedule
).result()
```
eliminar programación,primero debe deshabilitarla:
```python
ml_client.schedules.begin_disable(name=schedule_name).result()
ml_client.schedules.begin_delete(name=schedule_name).result()
```


