FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Instalar paquetes Python
RUN pip install --no-cache-dir \
    scikit-learn==1.5.0 \
    pandas==2.0.3 \
    numpy==1.24.3 \
    mlflow==2.8.1 \
    mltable==1.6.0 \
    azureml-mlflow \
    && rm -rf /root/.cache/pip

# Verificar instalación
RUN python -c "import sklearn, pandas, numpy, mlflow, mltable; print('Todos los paquetes instalados correctamente')"