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

