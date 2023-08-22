import time
import warnings

def fit_X(generator, preprocess_X):
    for i, var in enumerate(generator.input_variables):
        t = time.time()
        if var in preprocess_X:
            print(var)
            print(f"Preprocessing {var} - {i} \t | {time.time() - t}")
            preprocess_X[var].fit(generator.X[:, :, i])   
        else:
            print(f'Not fitting {var}')
    return preprocess_X 
    
def fit_Y(generator, preprocess_Y):
    for i, var in enumerate(generator.output_variables):
        t = time.time()
        if var in preprocess_Y:
            print(var)
            print(f"Preprocessing {var} - {i} \t | {time.time() - t}")
            preprocess_Y[var].fit(generator.Y[:, :, i])   
        else:
            print(f'Not fitting {var}')
    return preprocess_Y