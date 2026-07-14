# INDICE
- Defina un espacio de búsqueda de hiperparámetros.
- Configure el muestreo de hiperparámetros.
- Selección de una directiva de terminación anticipada.
- Ejecute un trabajo de barrido.

# Espacios de Búsqueda para Hiperparámetros

Guía esquemática de los tipos de valores para la optimización de hiperparámetros (Azure ML / Hyperopt), clasificados por su naturaleza matemática.

---

## Hiperparámetros

Los **hiperparámetros** son las variables de configuración externa que definen cómo aprende el algoritmo. A diferencia de los parámetros internos (los pesos que el modelo calcula por sí mismo), estas "perillas de ajuste" las decide el programador antes de iniciar el entrenamiento.

* **Tasa de Regularización** *(Algoritmos lineales como Regresión Logística)*
    * **Función:** Controla la complejidad del modelo penalizando las reglas demasiado rebuscadas.
    * **Objetivo:** Evitar el **sobreajuste** (*overfitting*), impidiendo que el modelo memorice el ruido de los datos de entrenamiento y asegurando que se adapte bien a datos nuevos.


* **Velocidad de Aprendizaje / *Learning Rate*** *(Técnicas de Aprendizaje Profundo como CNN)*
    * **Función:** Determina el tamaño del paso matemático que da el algoritmo al corregir sus errores y ajustar sus ponderaciones.
    * **Impacto:**
        * *Muy alta:* Pasos gigantescos que pueden saltarse la solución óptima (el modelo se vuelve inestable).
        * *Muy baja:* Pasos milimétricos que hacen el entrenamiento extremadamente lento o lo estancan.


* **Tamaño del Lote / *Batch Size*** *(Técnicas de Aprendizaje Profundo como CNN)*
    * **Función:** Define la cantidad de muestras de datos que el modelo procesa antes de actualizar sus pesos internos.
    * **Dinámica:** En lugar de analizar todo el dataset de golpe (lo que saturaría la memoria) o muestra por muestra (lo que sería ineficiente), se agrupan los datos (por ejemplo, en bloques de 32 o 64 elementos) para optimizar el rendimiento computacional.

---

## Hiperparámetros Discretos
Toman valores de un conjunto finito o forzados a una rejilla específica mediante cuantización ($q$).

### **Choice**
*   **Definición:** Selecciona un valor explícito de una lista definida.
*   **Ejemplo:** `Choice(values=[16, 32, 64, 128])` *(Típico para batch size)*
*   **Resultados posibles:** `16`, `32`, `64`, `128`

### **QUniform(min_value, max_value, q)**
*   **Definición:** Valores distribuidos uniformemente, redondeados al múltiplo de $q$ más cercano.
*   **Ejemplo:** `QUniform(min_value=10, max_value=50, q=10)` *(Número de árboles en Random Forest)*
*   **Resultados posibles:** `10`, `20`, `30`, `40`, `50`

### **QLogUniform(min_value, max_value, q)**
*   **Definición:** Distribución logarítmica (más densa en valores pequeños), redondeada a múltiplos de $q$.
*   **Ejemplo:** `QLogUniform(min_value=10, max_value=1000, q=10)`
*   **Resultados posibles:** `10`, `30`, `120`, `450`, `980`

### **QNormal(mu, sigma, q)**
*   **Definición:** Distribución normal en torno a la media ($\mu$), redondeada a múltiplos de $q$.
*   **Ejemplo:** `QNormal(mu=100, sigma=15, q=5)`
*   **Resultados posibles:** `95`, `100`, `105`, `85`, `115`

### **QLogNormal(mu, sigma, q)**
*   **Definición:** Distribución log-normal (sesgada a la derecha), redondeada a múltiplos de $q$.
*   **Ejemplo:** `QLogNormal(mu=0, sigma=1, q=2)`
*   **Resultados posibles:** `2`, `4`, `8`, `26`

---

## Hiperparámetros Continuos
Toman cualquier valor real dentro de una escala continua (infinitas posibilidades decimales).

### **Uniform(min_value, max_value)**
*   **Definición:** Todos los valores reales del rango tienen la misma probabilidad.
*   **Ejemplo:** `Uniform(min_value=0.0, max_value=1.0)` *(Tasa de dropout)*
*   **Resultados posibles:** `0.2341`, `0.7892`, `0.5000`

### **LogUniform(min_value, max_value)**
*   **Definición:** Distribución continua óptima para rangos que varían por órdenes de magnitud.
*   **Ejemplo:** `LogUniform(min_value=0.0001, max_value=0.1)` *(Learning rate)*
*   **Resultados posibles:** Alta probabilidad de valores pequeños como `0.0003` o `0.002`

### **Normal(mu, sigma)**
*   **Definición:** Valores reales bajo una curva de Gauss simétrica.
*   **Ejemplo:** `Normal(mu=0.0, sigma=0.1)` *(Inicialización de pesos / penalizaciones)*
*   **Resultados posibles:** `0.0123`, `-0.0845`, `0.1104`

### **LogNormal(mu, sigma)**
*   **Definición:** Valores reales siempre positivos con una cola larga hacia la derecha.
*   **Ejemplo:** `LogNormal(mu=0, sigma=1)` *(Regularización C en SVM o L2)*
*   **Resultados posibles:** `0.3452`, `1.2311`, `5.6742`

---

## DEFINICION DE ESPACIO DE BUSQUEDA DE HIPERPARAMETROS
En AzureML, un **espacio de búsqueda** es un conjunto de hiperparámetros definidos por el usuario, cada uno con su tipo de distribución (discreta o continua) y rango de valores. Este espacio guía al algoritmo de optimización para explorar combinaciones de hiperparámetros y encontrar la configuración que maximice el rendimiento del modelo.

```python
from azure.ai.ml.sweep import Choice, Normal

command_job_for_sweep = job(
    batch_size=Choice(values=[16, 32, 64]),    
    learning_rate=Normal(mu=10, sigma=3),
)
```

Aquí tienes el desglose esquemático de los conceptos para que de un vistazo entiendas qué es cada cosa y cuándo aplicarla en tus proyectos de Azure Machine Learning.

---

## 1. Método de Muestreo (Sampling Method)

Es la **estrategia o algoritmo** que decide cómo seleccionar puntos específicos dentro de un espacio de búsqueda lleno de opciones. En lugar de elegir valores a ciegas, define las "reglas del juego" para tomar muestras de hiperparámetros.

### Los 4 Tipos y Cuándo Usar Cada Uno

* **Muestreo de Cuadrícula (Grid Sampling)**
    * **Qué hace:** Realiza una búsqueda exhaustiva probando **todas y cada una** de las combinaciones posibles del espacio discreto.
    * **Cuándo se usa:**
        * Cuando el espacio de búsqueda es pequeño y tienes pocas variables.
        * Cuando los hiperparámetros son de tipo categórico o valores discretos fijos (ej. `[10, 20, 30]`).
        * Si el presupuesto de cómputo no es un problema y quieres asegurar que no se escape nada.




* **Muestreo Aleatorio (Random Sampling)**
    * **Qué hace:** Selecciona valores de forma **completamente al azar** en todo el espacio. Admite tanto valores discretos como continuos (rangos).
    * **Cuándo se usa:**
        * Como primera aproximación en un problema nuevo (línea base).
        * Cuando el espacio de búsqueda es enorme o continuo (ej. cualquier valor entre $0.001$ y $0.1$).
        * Si quieres ahorrar tiempo y dinero, ya que suele encontrar un rendimiento excelente con una fracción de los recursos de la cuadrícula.




* **Muestreo Sobol**
    * **Qué hace:** Una variante avanzada del muestreo aleatorio. Utiliza una secuencia matemática (secuencia cuasi-aleatoria de Sobol) para esparcir los puntos de forma **mucho más uniforme** por el espacio sin alinearlos en una cuadrícula rígida, y añade una semilla para que los resultados sean **100% reproducibles**.
    * **Cuándo se usa:**
        * Cuando quieres los beneficios de velocidad del muestreo aleatorio pero evitando que los puntos queden "amontonados" por azar en una misma zona.
        * Esencial si necesitas reproducibilidad estricta en tus experimentos científicos o auditorías de modelos.




* **Muestreo Bayesiano (Bayesian Sampling)**
    * **Qué hace:** Prueba un par de combinaciones al azar, ve el resultado (si el modelo mejora o empeora) y, basándose en la probabilidad, elige la siguiente combinación donde cree que hay más opciones de éxito.
    * **Cuándo se usa:**
        * Cuando el presupuesto de cómputo es muy limitado pero buscas la máxima optimización posible.
        * Cuando entrenar el modelo es muy costoso en tiempo y necesitas que cada intento cuente.
        * *Restricción:* No permite lanzar todos los entrenamientos en paralelo, ya que el paso B necesita saber obligatoriamente qué tal le fue al paso A.





---

## 2. Trabajo de Barrido (Sweep Job)

Es la **infraestructura y ejecución física** en la nube. Es la orden que le das a Azure Machine Learning para que ponga a trabajar los recursos de computación.

* **Función principal:** Actúa como el "orquestador". Toma el Espacio de Búsqueda y el Método de Muestreo elegidos, configura las máquinas virtuales, levanta los contenedores y ejecuta los scripts de entrenamiento en paralelo o secuencialmente.
* **Gestión de recursos:** Controla las políticas de terminación anticipada (por ejemplo, si un entrenamiento va muy mal, el *Sweep Job* lo cancela a la mitad para no gastar dinero).
* **Resultado final:** Al terminar, recopila las métricas de todos los modelos entrenados y te devuelve el "Modelo Campeón" con los mejores hiperparámetros encontrados.

> **En una frase:** El *Método de Muestreo* es el cerebro matemático (el algoritmo de selección) y el *Trabajo de Barrido* es el músculo operativo (la ejecución en Azure).


```python
from azure.ai.ml.sweep import Normal, Uniform
# from azure.ai.ml.sweep import RandomSamplingAlgorithm
# si usas sobol


command_job_for_sweep = command_job(
    batch_size=Choice(values=[16, 32, 64]),   
    learning_rate=Normal(mu=10, sigma=3),
)

sweep_job = command_job_for_sweep.sweep(
    sampling_algorithm = "random",
    # tambien puede ser "grid", "bayesian", o
    # RandomSamplingAlgorithm(seed=123, rule="sobol")
    ...
)
```

Terminación del trabajo de barrido
Cuando lanzas un trabajo de barrido, puedes definir **políticas de terminación anticipada** para ahorrar tiempo y dinero. 
Se usa:
- Para cuando un modelo no está rindiendo bien
- Para cuando ya tenemos un resultado suficientemente bueno

Para detener un trabajo de barrido en función del rendimiento de los modelos, puede usar una directiva de terminación anticipada.

### Directiva de terminación anticipada
- beneficiosa cuando se trabaja con 
    - hiperparámetros continuos en el espacio de búsqueda
    - método de muestreo aleatorio o bayesiano.
- Hay dos parámetros principales al elegir usar una directiva de terminación anticipada:
    - evaluation_interval: especifica en qué intervalo desea que se evalúe la directiva. 
    - delay_evaluation: especifica cuándo empezar a evaluar la directiva. Este parámetro permite que se completen al menos un mínimo de pruebas sin una directiva de terminación anticipada que les afecte.
- hay tres opciones para la finalización anticipada:
    - **Política de bandido (Bandit Policy):** 
        - usa un slack_factor (relativo) o slack_amount (absoluto). 
        - Cualquier nuevo modelo debe desempeñarse dentro del margen de tolerancia del modelo de mejor rendimiento.
        - Por ejemplo, el siguiente código aplica una directiva de ladrón con un retraso de cinco pruebas, evalúa la directiva en cada intervalo y permite una cantidad de holgura absoluta de 0,2.

        ```Python
        from azure.ai.ml.sweep import BanditPolicy

        sweep_job.early_termination = BanditPolicy(
            slack_amount = 0.2, 
            delay_evaluation = 5, 
            evaluation_interval = 1
        )
        ```

    - **Directiva de detención de mediana (Median Stopping Policy):** 
        - Usa la mediana de los promedios de la métrica principal. z
        - Cualquier nuevo modelo debe funcionar mejor que la mediana.
        - Por ejemplo, el código siguiente aplica una directiva de detención mediana con un retraso de cinco pruebas y evalúa la directiva a cada intervalo.

        ```Python
            from azure.ai.ml.sweep import MedianStoppingPolicy

            sweep_job.early_termination = MedianStoppingPolicy(
                delay_evaluation = 5, 
                evaluation_interval = 1
            )
        ```
    - **Directiva de selección de truncamiento (Truncation Selection Policy):** 
        - Usa un truncation_percentage, que es el porcentaje de pruebas con menor rendimiento. Cualquier modelo nuevo debe funcionar mejor que las pruebas de menor rendimiento.

        - Por ejemplo, el código siguiente aplica una directiva de selección de truncamiento con un retraso de cuatro pruebas, evalúa la directiva en cada intervalo y usa un porcentaje de truncamiento del 20 %.

        ```Python
        from azure.ai.ml.sweep import TruncationSelectionPolicy

        sweep_job.early_termination = TruncationSelectionPolicy(
            evaluation_interval=1, 
            truncation_percentage=20, 
            delay_evaluation=4 
        )
        ```
### script de entrenamiento para el ajuste de hiperparámetros
Para ejecutar un trabajo de barrido.

Incluya un argumento para cada hiperparámetros que desee variar.
Registrar la métrica de rendimiento de destino con MLflow. 
Una métrica registrada permite al trabajo de barrido evaluar el rendimiento de las pruebas que inicia e identificar el que genera el modelo de mejor rendimiento.

EJEMPLO:

```python
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import mlflow

# get regularization hyperparameter
parser = argparse.ArgumentParser()
parser.add_argument('--regularization', type=float, dest='reg_rate', default=0.01)
args = parser.parse_args()
reg = args.reg_rate

# load the training dataset
data = pd.read_csv("/data/iris-data/irish.csv")

# separate features and labels, and split for training/validatiom
X = data[['feature1','feature2','feature3','feature4']].values
y = data['label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# train a logistic regression model with the reg hyperparameter
model = LogisticRegression(C=1/reg, solver="liblinear").fit(X_train, y_train)

# calculate and log accuracy
y_hat = model.predict(X_test)
acc = np.average(y_hat == y_test)
mlflow.log_metric("Accuracy", acc)