import matplotlib.pyplot as plt
import numpy as np

# Given an 

def plot_one_profile_column(data, variables, x, y, n_col=5):
    """
    data : xarray of either input or output datarray must have 4 dimensions (_, _, <x, <y)
    variables : list of string (key of variables to plot)
    x : index to plot
    y : index to plot
    """
    N = len(variables)
    n_row = (N-1)//n_col + 1 
    f = plt.figure( figsize=(n_col*4, n_row*7))
    f.suptitle(f"Coordinates : {x}-{y}")
    for n, var in enumerate(variables):
        ax = f.add_subplot(n_row,n_col,n+1)
        plot = data[var][0, :, x, y]
        ax.plot(plot, np.arange(len(plot)))
        ax.set_title(var)
    return f

def plot_one_profile_surface(data, variables, z, n_col=4):
    """
    data : xarray of either input or output datarray must have 4 dimensions (_, z, _, _)
    variables : list of string (key of variables to plot)
    zi : level index 
    """
    N = len(variables)
    n_row = (N-1)//n_col + 1 
    f = plt.figure( figsize=(n_col*4, n_row*7))
    f.suptitle(f"Coordinates : {z}")
    for n, var in enumerate(variables):
        ax = f.add_subplot(n_row,n_col,n+1)
        if len(data[var].shape)==4:
            plot = data[var][0, z, :, :]
        else:
            plot = data[var][0, :, :]            
        ax.imshow(plot)
        ax.set_title(var)
    return f

# ANIMATION :
from PIL import Image
from matplotlib.animation import FuncAnimation

def update_hist(frame, datas):
    plt.hist(datas[frame], bins=20)
    plt.title(f"{frame}")
    plt.axis('off')

def update_imshow(frame, datas):
    plt.imshow(datas[frame])
    plt.title(f"{frame}")
    plt.axis('off')

def animate_plane(ds, var, filepath=None):
    data = []
    for z in range(72):
        data.append(ds[var].values[0, z, ::5, ::5])  
        # Create the animation using FuncAnimation
    fig = plt.figure()
    lbd_update = lambda x: update_imshow(x, datas=data)
    anim = FuncAnimation(fig, lbd_update, frames=len(data), interval=10)
    # Save the animation as a gif
    with Image.new('RGB', (1, 1)) as img:
        anim.save(filename=filepath, writer='pillow')
    # Show the animation
    plt.show()
    return data

def animate_plane_delta(ds, var, filepath):
    data = []
    for z in range(72):
        data.append(ds[var].values[0, 1+z, ::5, ::5] -  ds[var].values[0, z, ::5, ::5])  
        # Create the animation using FuncAnimation
    fig = plt.figure()
    lbd_update = lambda x: update_imshow(x, datas=data)
    anim = FuncAnimation(fig, lbd_update, frames=len(data), interval=10)
    # Save the animation as a gif
    with Image.new('RGB', (1, 1)) as img:
        anim.save(filename=filepath, writer='pillow')
    # Show the animation
    plt.show()
    return data


# EARTH PLOT :
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def _separate_faces(data):
    width, height = data.shape
    face_1 = np.rot90(data[:width//6].T)
    face_2 = np.rot90(data[width//6:2*width//6].T)
    face_3 = np.rot90(data[2*width//6:3*width//6].T)
    face_4 = data[3*width//6:4*width//6].T
    face_5 = data[4*width//6:5*width//6].T
    face_6 = data[5*width//6:6*width//6].T
    faces= [face_1, face_2,face_3,face_4,face_5,face_6 ]
    return faces, height, width

def plot_square_earth(data):
    """Plot the earth cube (two different views)

    Args:
        data (np,array): array of a netcdf file. the 6 faces of the cubes are separated then plot, must be two D

    Returns:
        plt.fig : the figure 
    """

    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    faces, height, width = _separate_faces(data)
    # Define the coordinates for the vertices of the image plane
    prec = 10
    x = np.linspace(-1, 1, height//prec)
    y = np.linspace(-1, 1, height//prec)
    X, Y = np.meshgrid(x, y)

    # Norther Em
    ax1.plot_surface(X, Y, 1+np.zeros_like(X), facecolors=plt.cm.viridis(faces[2][::prec, ::prec]), rstride=1, cstride=1, alpha=0.2)
    # Afri
    ax1.plot_surface(1 + np.zeros_like(X),X, -Y,  facecolors=plt.cm.viridis(faces[0][::prec, ::prec]), rstride=1, cstride=1, alpha=0.2)
    # Oceania
    ax1.plot_surface(X, -1+np.zeros_like(X), -Y, facecolors=plt.cm.viridis(faces[4][::prec, ::prec]), rstride=1, cstride=1, alpha=0.2)

    # Antartica
    ax2.plot_surface(-X, -Y, 1+np.zeros_like(X), facecolors=plt.cm.viridis(faces[5][::prec, ::prec]), rstride=1, cstride=1, alpha=0.2)
    # Oceania
    ax2.plot_surface(-X,  -1+np.zeros_like(X), Y,facecolors=plt.cm.viridis(faces[3][::prec, ::prec]), rstride=1, cstride=1, alpha=0.2)
    # Asia
    ax2.plot_surface(1 + np.zeros_like(X),-X, Y, facecolors=plt.cm.viridis(faces[1][::prec, ::prec]), rstride=1, cstride=1, alpha=0.2)

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    return fig