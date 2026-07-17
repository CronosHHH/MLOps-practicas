# import libraries
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

# setup arg parser
parser = argparse.ArgumentParser()

# add arguments
parser.add_argument("--input_data", dest='input_data', type=str)
parser.add_argument("--output_data", dest='output_data', type=str)

# parse args
args = parser.parse_args()

# read the data
df = pd.read_csv(args.input_data)

# remove missing values
df = df.dropna()

# normalize the data    
scaler = MinMaxScaler()
num_cols = ['feature1','feature2','feature3','feature4']
df[num_cols] = scaler.fit_transform(df[num_cols])

# save the data as a csv
# se guarda el archivo de salida en la ruta especificada por el 
# argumento output_data, con el nombre "prepped-data.csv"
output_df = df.to_csv(
    (Path(args.output_data) / "prepped-data.csv"), 
    index = False
)
