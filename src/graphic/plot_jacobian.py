import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset


# Use input
# ---------
profile = 'i0720_j0720'
path = ''

fields3d = ['dflxdpl','dflxdt','dflxdq','dflxdqi','dflxdql','dflxdri','dflxdrl',
            'dflxdo3','dflxdfcld',
            'dfdtsdpl','dfdtsdt','dfdtsdq','dfdtsdqi','dfdtsdql','dfdtsdri','dfdtsdrl',
            'dfdtsdo3','dfdtsdfcld']

fields2d = ['dflxdts','dflxdemis','dfdtsdts','dfdtsdemis']



# Open NetCDF file for reading

fh = Dataset(path+'jacobian_'+profile+'.nc4')

z = np.arange(1, 73, 1)


# Contour plots of 3D fields
# --------------------------
for n in range(len(fields3d)):

  #Get Jabobian component
  varname = fields3d[n]

  print(varname)

  field = fh.variables[varname][:]
  print(field.shape)

  maxf = np.max(np.abs(field))
  if maxf == 0:
    maxf = 1.0
  incf = 2*maxf/51.
  clevs = np.arange(-maxf,maxf+incf,incf)

  fig, ax = plt.subplots()
  im = ax.contourf(z, z, np.transpose(field), clevs, cmap = 'PRGn')
  ax.set_aspect('equal', 'box')
  ax.set_title(varname)
  ax.invert_yaxis()
  fig.colorbar(im,ax=ax)

  plt.savefig(path+'jacobian_'+varname+'_'+profile+'.png')

  plt.close()

# Line plots of 2D fields
# -----------------------
for n in range(len(fields2d)):

  #Get Jabobian component
  varname = fields2d[n]

  print(varname)

  field = fh.variables[varname][:]

  fig, ax = plt.subplots()
  im = ax.plot(field, z)
  ax.set_title(varname)
  ax.invert_yaxis()
  plt.ylim([72, 1])

  plt.savefig(path+'jacobian_'+varname+'_'+profile+'.png')

  plt.close()


fh.close()
