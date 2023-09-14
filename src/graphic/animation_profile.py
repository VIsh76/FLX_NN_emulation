from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from utils import deconstruct_cube, reshaper


def show_one_time(t, data_loader, var, lev, is_in_x):
    """Will iterate over time

    Args:
        t (int): id of the loader
        data_loader (_type_): _description_
        var (str): variable to plot
        lev (int): level to plot
        is_in_x (bool): indication if var is in input_variable or output_variable
    """
    x, y = data_loader[t]
    if is_in_x:
        id_var = data_loader.input_variables.index(var)
        cube_face = reshaper(x[:, id_var, lev].to('cpu').numpy() )
    else:
        id_var = data_loader.output_variables.index(var)
        cube_face = reshaper(y[:, id_var, lev].to('cpu').numpy())
        
    flat_face = deconstruct_cube(cube_face)
    plt.imshow(flat_face)
    plt.title(f'Time {t}, var {var}, level {lev}')

def show_one_lev(t, data_loader, var, lev, is_in_x):
    """Will iterate over levels

    Args:
        t (int): level to plot
        data_loader (_type_): _description_
        var (str): variable to plot
        lev (int): d of the loader
        is_in_x (bool): indication if var is in input_variable or output_variable
    """
    show_one_time(lev, data_loader, var, t, is_in_x)
 
 
def loss_over_time(t, data_loader, var, lev, model, use_mean=False):
    x, y = data_loader[t]
    id_var = data_loader.output_variables.index(var)
    y_pred = model(x).to('cpu').numpy()
    error = (y.to('cpu').numpy() - y_pred)**2
    if use_mean:
        error = np.mean(error[:, id_var], axis=-1)
    else:
        error = error[:, id_var, lev]    

    cube_face = reshaper(error)   
    flat_face = deconstruct_cube(cube_face)
    plt.imshow(flat_face)
    plt.title(f'Error time {t}, var {var}, level {lev}')


def loss_over_levels(lev, data_loader, var, t, model, use_mean=False):
    x, y = data_loader[t]
    id_var = data_loader.output_variables.index(var)
    y_pred = model(x).to('cpu').numpy()
    error = (y.to('cpu').numpy() - y_pred)**2
    if use_mean:
        error = np.mean(error[:, id_var], axis=-1)
    else:
        error = error[:, id_var, lev]    

    cube_face = reshaper(error)   
    flat_face = deconstruct_cube(cube_face)
    plt.imshow(flat_face)
    plt.title(f'Error time {t}, var {var}, level {lev}')

  
def animate_var_over_time(data_loader, updater_fct, figsize, length, filepath, **kwargs):
    fig = plt.figure(figsize=figsize)
    lbd_update = lambda x: updater_fct(x, data_loader, **kwargs)
    anim = FuncAnimation(fig, lbd_update, frames=len(length), interval=10)
    # Save the animation as a gif
    with Image.new('RGB', (1, 1)) as img:
        anim.save(filename=filepath, writer='pillow')
    # Show the animation
    plt.show();
