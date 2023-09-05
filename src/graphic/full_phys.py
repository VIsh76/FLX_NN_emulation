import matplotlib.pyplot as plt
import numpy as np

def show_all_faces(X, id_lev, id_var, resolution, var=''):
    f = plt.figure()
    for i in range(6):
        ax = f.add_subplot(2, 3, i+1)
        ax.imshow( np.reshape(X[:, id_lev, id_var], (resolution, resolution, 6))[:, :, i]  )
    f.suptitle(f'{var} : {id_lev} | all faces')
    return f
