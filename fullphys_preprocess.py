import matplotlib.pyplot as plt
import numpy as np
import datetime
import pickle
import time
import os

# Parameters :
from params import input_variables, pred_variables
from params import preprocess_X, preprocess_Y, preprocess_X_path, preprocess_Y_path
from params import data_path, graph_path, output_path

# Additioal params:
from params import nb_portion, save, test

print("Creating folders")
os.makedirs(output_path, exist_ok=True)
os.makedirs(graph_path, exist_ok=True)


in_channel = len(input_variables)
out_channel = len(pred_variables)

from src.load_data.generator import PhysicGenerator
from src.preprocess.fit_preprocess import fit_X, fit_Y
from src.graphic.preprocessor import plot_series

if __name__ == '__main__':
    ################################# FIT X
    PG_ex = PhysicGenerator(data_path = data_path,
                       nb_portions = nb_portion,
                       batch_size = 1,
                       input_variables = input_variables,
                       output_variables = pred_variables,
                       shuffle=False,
                       verbose=1)    
    print('fit Preprocess X')
    fit_X(PG_ex, preprocess_X)

    ################################# FIT Y
    PG_ex = PhysicGenerator(data_path = data_path,
                       nb_portions = nb_portion,
                       batch_size = 1,
                       input_variables = pred_variables,
                       output_variables = pred_variables,
                       shuffle=False,
                       verbose=1)    
    print('Preprocess Y')
    fit_Y(PG_ex, preprocess_Y )

    ################################# RESET :
    PG_ex = PhysicGenerator(data_path = data_path,
                       nb_portions = nb_portion,
                       batch_size = 1,
                       input_variables = input_variables,
                       output_variables = pred_variables,
                       shuffle=False,
                       verbose=1)
    
    print('---- input ----')
    X_prepross = np.zeros_like(PG_ex.X)
    for i, var in enumerate(PG_ex.input_variables):
        if var in preprocess_X:
            print(f"Call {var} - {i}")
            X_prepross[:,:,i] = preprocess_X[var](PG_ex.X[:, :, i].copy())

    print('---- output ----')
    Y_prepross = np.zeros_like(PG_ex.Y)
    for i, var in enumerate(PG_ex.output_variables):
        if var in preprocess_Y:
            print(f"Call {var} - {i}")
            Y_prepross[:,:,i] = preprocess_Y[var](PG_ex.Y[:, :, i].copy())

# %%
    # SAVE :
    if save:
        f = plot_series(input_variables, PG_ex.X,      nb_plot=10, graph_path=graph_path, title = 'input_orig', save=save)
        f = plot_series(input_variables, X_prepross, nb_plot=10, graph_path=graph_path, title='input_preprop', save=save)
        f = plot_series(pred_variables,  PG_ex.Y,      nb_plot=10, graph_path=graph_path, title='pred_orig', save=save)
        f = plot_series(pred_variables,  Y_prepross, nb_plot=10, graph_path=graph_path, title='pred_preprop', save=save)
        print("saving")
        path = f'{preprocess_X_path}'
        with open(path, 'wb') as handle:
            pickle.dump(preprocess_X, handle, protocol=pickle.HIGHEST_PROTOCOL)
        path = f'{preprocess_Y_path}'
        with open(path, 'wb') as handle:
            pickle.dump(preprocess_Y, handle, protocol=pickle.HIGHEST_PROTOCOL)