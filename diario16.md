Vamos a controlar la plataforma de AzureML desde un script de Python, a traves del SDK de Python, para ello vamos a usar la libreria `azure-ai-ml` que nos permite interactuar con los recursos de AzureML.

- se configura el experimento mediante la clase automl. 
- Para la clasificación, use la función: automl.classification como se muestra en el ejemplo siguiente:

```python
from azure.ai.ml import automl

# 1. Configuras el experimento de Clasificación
classification_job = automl.classification(
    compute="mi-cluster-de-computo",      # La máquina virtual donde correrá
    experiment_name="auto-ml-vias",
    training_data=my_training_data_input,  # El recurso MLTable que configuraste antes
    target_column_name="target",           # La columna que quieres clasificar
    primary_metric="accuracy"              # Cómo vas a medir al mejor modelo
)

# 2. Configuras los límites (para que no gaste dinero infinito)
classification_job.set_limits(
    timeout_minutes=30,
    max_trials=5
)
```

### Especificación de la métrica principal

`primary_metric` -> es la métrica que se utilizará para clasificar todos los modelos entrenados y seleccionar el mejor. 

Azure Machine Learning admite un conjunto de métricas con nombre para cada tipo de tarea.
```python
from azure.ai.ml.automl import ClassificationPrimaryMetrics
 
list(ClassificationPrimaryMetrics)
```
### Establecimiento de los límites
`set_limits()` -> establece los límites de tiempo y número de modelos a entrenar. 
Esto es importante para controlar el gasto en la nube y evitar que el experimento se ejecute indefinidamente.
```python
classification_job.set_limits(
    timeout_minutes=60, 
    trial_timeout_minutes=20, 
    max_trials=5,
    enable_early_termination=True,
)
```
- timeout_minutes: número de minutos después de que finalice el experimento autoML completo.
- trial_timeout_minutes: número máximo de minutos que puede tardar una prueba.
- max_trials: número máximo de pruebas o modelos entrenados.
- enable_early_termination: indica si finalizar el experimento si la puntuación no mejora a corto plazo.
- max_concurrent_trials: número máximo de pruebas que se pueden ejecutar al mismo tiempo. Esto es útil si tienes un clúster de computación con varios nodos y quieres aprovecharlos al máximo. (compuracion en paralelo)

si ya sabes que algoritmos no te interesan, puedes excluirlos de la búsqueda con `exclude_algorithms`:
```python
classification_job.set_limits(
    exclude_algorithms=["LightGBM", "XGBoost"]
)
```
---
## Enviado del experimento a AzureML

```python
# submit the AutoML job
returned_job = ml_client.jobs.create_or_update(
    classification_job
)       
```

para que te salga en AzureML Studio, tienes que ir a la sección de Experimentos y buscar el nombre que le pusiste al experimento (en este caso "auto-ml-vias"). 
Allí podrás ver el progreso del entrenamiento, los modelos generados y sus métricas de rendimiento.
```python
aml_url = returned_job.studio_url
print("Monitor your job at", aml_url)
```
---

# Evaluación y comparación de modelos
dentro de AzureML Studio, una vez que el experimento ha finalizado, puedes ver todos los modelos entrenados y sus métricas.

en la sección de "Modelos" puedes comparar los modelos generados, ver sus métricas y elegir el mejor para desplegarlo en producción.

## Explorar los pasos de preprocesamiento

### Data Guardrails - Limites de Proteccion de Datos
ANTES de entrenar el modelo.

Al activar la "caracterización" o "featurization" en AutoML, AzureML aplica automáticamente una serie de transformaciones y protecciones a los datos de entrada.
Es como un *chequeo medico* a los datos.
Se enfoca en trés puntos críticos:

- Deteccion del equilibrio de clases
 - comprueba si el dataset de entrenamiento tiene un balance adecuado entre las clases de la columna objetivo.
 - Si hay un desbalance significativo, AutoML puede aplicar técnicas de remuestreo para equilibrar las clases y evitar que el modelo se sesgue hacia la clase mayoritaria.
 - Ejemplo: sale siempre "no fraude" porque hay 99.9999% de transacciones legítimas y solo 0.0001% de fraude.

- imputación de valores faltantes
 - AutoML detecta automáticamente si hay valores faltantes en las columnas (NaN o null) y aplica técnicas de imputación para rellenarlos.

- detección de características de alta cardinalidad
 - AutoML identifica columnas categóricas con un número muy alto de categorías únicas (alta cardinalidad) y aplica técnicas de codificación adecuadas para evitar problemas de dimensionalidad y sobreajuste del modelo (evitar o reducir el "overfitting").

AzureML nos mostrata 3 estados (verde, amarillo y rojo)

# Escalado, Normalización y Transformación de Datos
DESPUES de entrenar el modelo.

Cómo leer la lista de resultados una vez que AutoML ha terminado de probar modelos. 
Azure no solo prueba algoritmos, sino que prueba combinaciones de "Preprocesador + Algoritmo".

En la tabla de modelos verás nombres compuestos en la columna Nombre del algoritmo, estructurados como [Técnica de Escalado], [Algoritmo]:

- MaxAbsScaler, LightGBM Significa que para ese modelo concreto, 
 - Azure primero aplicó la técnica MaxAbsScaler (que toma cada número de la columna y lo divide por el valor absoluto máximo de esa misma columna, logrando que todos los datos queden encajados estrictamente en un rango entre -1 y 1). 
 - Una vez escalados los datos, se los entregó al algoritmo de clasificación LightGBM (un modelo basado en árboles de decisión muy rápido y eficiente) para que se entrenara.

---

## Identificar y Comparar Modelos fácilmente
AzureML te hace la clasificacion de los modelos con mas accuracy, pero no es lo unico a tener en cuenta, en la vista-multimétrica puedes ver otras métricas como precision, recall, f1-score, AUC, etc. y compararlas para elegir el modelo que mejor se adapte a tus necesidades.
Tenemos:
- **Accuracy**: Proporción de predicciones correctas sobre el total de predicciones. 
  - Precision= True Positives / (True Positives + False Positives) -> Qué proporción de predicciones positivas fueron correctas.
  - Es útil cuando las clases están equilibradas.
  
- **Recall** (o sensibilidad o exhaustividad): Qué proporción de casos positivos reales fueron correctamente identificados por el modelo.
  - Recall= True Positives / (True Positives + False Negatives)
  - Es útil cuando las clases están desbalanceadas y queremos minimizar los falsos negativos.
_
- **F1-score**: Es la media armónica de precision y recall
  - F1-score= 2 * (Precision * Recall) / (Precision + Recall)
  - es util cuando queremos un balance entre precision y recall, especialmente en problemas de clasificación desbalanceada.

## Explicabilidad del Modelo (Interpretabilidad)
Los modelos complejos (como LightGBM o los ensamblados) suelen actuar como "cajas negras": hacen predicciones excelentes, pero es difícil saber por qué han tomado una decisión. Azure soluciona esto permitiéndote generar explicaciones del modelo:

- Modelo de mejor rendimiento: Puedes marcar una casilla al configurar AutoML para que, nada más terminar, calcule automáticamente las explicaciones del modelo ganador.

- Cualquier otro modelo: Si el segundo o tercer mejor modelo te llama la atención (por ejemplo, porque es un algoritmo más ligero o más rápido de ejecutar en producción), puedes seleccionarlo manualmente y hacer clic en "Explicar modelo" (Explain model).

- que te aporta esta función: Te generará gráficos de importancia de características (Feature Importance), permitiéndote ver exactamente qué columnas de tus datos influyeron más en las predicciones del modelo (por ejemplo, si para detectar fraudes el factor determinante fue la hora de la transacción o el país de origen).

## Pasar de la Automatización al Código
Si tras ver los resultados de AutoML quieres afinar el modelo al máximo haciendo cosas como:

- Diseñar una lógica de ingeniería de características muy específica de tu negocio (custom features).

- Ajustar manualmente los hiperparámetros del algoritmo a valores que AutoML no probó.

- Modificar la arquitectura de la red o del modelo.

Azure te permite exportar el código generado por AutoML a un Jupyter Notebook (cuaderno). De este modo, tomas el modelo que descubrió la automatización como tu nueva "línea base" (baseline) y continúas el desarrollo refinándolo a mano con código Python personalizado, integrándolo de forma limpia en tus repositorios y pipelines de producción.