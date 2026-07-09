ipynotebook y MLflow

# Definiciones

## MLflow

# Montaje

## Creo entorno local
```bash
python3 -m venv azure_mlflow_env
source azure_mlflow_env/bin/activate
*entramos en el entorno virtual*
pip install --upgrade pip
pip install mlflow azureml-mlflow azure-ai-ml azure-identity ipykernel
pip install python-dotenv //para la configuracion de variables de entorno
deactivate
*salir de entorno virtual*
az login
```

## Configuracion de MLflow para el seguimiento de experimentos

con MLflow vamos a registrar todo sobre el modelo que está entrenando:
- parametros
- metricas
- artefactos (modelos, imagenes, etc)

Asi compararemos las ejeciciones del notebook con los modelos de AutoML.

Para ello estableceremos AzureML como almacen de seguimiento.

--- 
---


# Entrenamiento y seguimiento de modelos en cuadernos


```python
import mlflow

mlflow.set_experiment(experiment_name="mi-experimento-local-test")
```
---

me dio problemas el:
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
porque el plugin azureml-mlflow ya configura la conexión automáticamente. Si se asigna manualmente, da conflicto con azureml_artifacts_builder.
  
---
  
# Evaluación de modelos con el panel de control de IA responsable  
Más [información](https://learn.microsoft.com/es-es/training/modules/experiment-azure-machine-learning/7-responsible-ai-dashboard?tabs=error-analysis)

la predicción del modelo, debe tener en cuenta los seis principios de inteligencia artificial responsable de Microsoft: Equidad, confiabilidad y seguridad, privacidad y protección de datos, inclusión, transparencia y responsabilidad.
### Creación de un panel de IA responsable
crear una *canalización* mediante los componentes RAI integrados de Azure Machine Learning. La canalización debe:

Comience con RAI Insights dashboard constructor.
Incluya uno o varios componentes de la herramienta RAI para la información que necesita.
Termine con Gather RAI Insights dashboard para recopilar todo en un panel.

Explora el panel 
tiene varias pestañas, cada una de las cuales proporciona información sobre un aspecto diferente del modelo. Por ejemplo, la pestaña de equidad muestra cómo se comporta el modelo en diferentes grupos demográficos, mientras que la pestaña de explicabilidad muestra qué características son más importantes para las predicciones del modelo.   
Tiene lo de la:
- distribución de errores
- explicaciones, que es un tipo de explicación que muestra qué características de entrada son más importantes para las predicciones del modelo. Por ejemplo, si el modelo predice que un cliente es propenso a abandonar un servicio, una explicación podría mostrar que la falta de interacción con el servicio es una característica importante para esa predicción.
- contrafactuales, que es un tipo de explicación que muestra cómo cambiarían las predicciones del modelo si se modificaran ciertas características de entrada. Por ejemplo, si el modelo predice que un cliente no es elegible para un préstamo, un contrafactual podría mostrar que si el cliente tuviera un ingreso más alto, el modelo lo consideraría elegible.
- analisis causales, que es un tipo de análisis que intenta identificar las relaciones causales entre las características de entrada y las predicciones del modelo. Por ejemplo, si el modelo predice que un cliente es propenso a abandonar un servicio, un análisis causal podría mostrar que la falta de interacción con el servicio es una causa importante de esa predicción.

