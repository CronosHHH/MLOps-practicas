import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import mlflow
import os

# PENSAR QUE TODO EL CODIGO DE ESTE SCRIPT SE EJECUTA EN LA NUBE
# , NO EN EL NOTEBOOK, POR LO QUE NO PODEMOS USAR VARIABLES DEL 
# NOTEBOOK, SOLO LAS QUE LE PASAMOS COMO ARGUMENTOS


# get regularization hyperparameter
parser = argparse.ArgumentParser()
parser.add_argument('--regularization', type=float, dest='reg_rate', default=0.01)
parser.add_argument('--training_data', type=str, dest='training_data')
# args = parser.parse_args([]) pq ya no estamos en el notebook
args = parser.parse_args()  # ← Lee los argumentos reales
reg = args.reg_rate 


# data = pd.read_csv("./data/iris-data/iris.csv") pasa a ser
data = pd.read_csv(os.path.join(args.training_data, "iris.csv")) 

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)


# Model
with mlflow.start_run():
    model = LogisticRegression(C=1/reg, solver="lbfgs", max_iter=1000).fit(X_train, y_train)

    # Metrics
    y_hat = model.predict(X_test)
    acc = np.average(y_hat == y_test)
    mlflow.log_metric("Accuracy", acc)