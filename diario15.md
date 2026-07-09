Vamos a crear un proyecto ML automatizado (*AutoML*)

- vamos a empezar con un *training Dataset*, que nos sirve para la etapa de experimentacion, no tiene porque ser todos los datos.

- para que AzureML entienda los datos, tenemos que enpaquetarlos en una carpeta que tenga 2 cosas:
1. datos fisicos csv, json, parquet ...
2. archivo llamado exactamente:
   - `MLTable` -> es un YAML, le dice a azure como debe interpretar, delimitar o transformar los datos, para convertirlos en un DataFrame.
 
### Después de crear el recurso de datos, puede especificarlo como entrada con el código siguiente: 
Una vez que esa estructura ya está registrada en la nube de Azure con un nombre (por ejemplo, input-data-automl) y una versión (1), ya no necesitas lidiar con rutas locales ni con configuraciones de lectura en tus scripts de Python. 

```python
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml import Input

# Aquí apuntas directamente al "marcador" o "acceso directo" en la nube de Azure
my_training_data_input = Input(
    type=AssetTypes.MLTABLE,               # Le indicas que la entrada es una estructura MLTable
    path="azureml:input-data-automl:1"     # El identificador único en Azure (nombre:versión)
)
```
### Una vez creado el recurso de datos, puede configurar el experimento de AutoML.
En Azure Machine Learning, AutoML (Automated Machine Learning) actúa como un "científico de datos automatizado". En lugar de que tú escribas manualmente el código para probar 20 algoritmos distintos y ajustar cientos de parámetros, tú solo configuras las reglas del experimento.

Al configurar el experimento, le defines a Azure cosas como:

- La tarea: En este caso, le vas a indicar que quieres resolver un problema de Clasificación (por ejemplo, predecir si una transacción es fraude o no, o si un cliente va a abandonar un servicio).

- La columna objetivo (target): Qué columna de tu tabla es la que el modelo debe aprender a predecir.

- Los datos de entrada: Aquí es donde le pasas el recurso de datos (my_training_data_input) que acabas de instanciar en el paso anterior.

- Los límites: Cuánto tiempo máximo puede ejecutar el experimento o cuántos modelos distintos quieres que pruebe como límite.

### Ultimo pasos antes de entrenar el modelo.
Antes de iniciar el entrenamiento, autoML ejecuta de forma automática (o bajo tu configuración) una fase de preprocesamiento y featurization (caracterización). Esto incluye transformaciones críticas como:

- Limpieza de valores faltantes (Imputación): Si en la tabla de datos faltan edades o ingresos de algunos usuarios, AutoML los rellena automáticamente usando la media o el valor más frecuente para que el algoritmo no falle.

- Codificación categórica (Encoding): Transforma las variables de texto (por ejemplo: ciudades como "Granada" o "Málaga") en números (como 0 y 1) para que los modelos matemáticos puedan operar con ellas.

- Escalado de características (Scaling): Si una columna tiene valores entre 0 y 1 (por ejemplo, una tasa de conversión) y otra tiene valores entre 10,000 y 100,000 (como un salario), AutoML las normaliza para que las cifras gigantescas no eclipsen a las pequeñas durante el entrenamiento.

- Ingeniería de características de texto o fechas: Si tienes una columna con fechas completas, AutoML puede desglosarla automáticamente en "día de la semana", "mes" o "si era festivo", aportando mucho más valor al modelo de clasificación.

Rsumen:
- tenemos datoset entrenamiento con la estructura prefijada
- creamos un recurso de datos en AzureML
- le definimos la tarea de clasificación, la columna objetivo y los límites del experimento
- nos limpia y transforma los datos automáticamente para que el modelo pueda aprender de ellos sin problemas.

* AutoML aplica escalado y normalizacion automaticamente.
