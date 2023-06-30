import numpy as np
import matplotlib.pyplot as plt

# Plot output of the FOrtran model

def Plot(L):
    fig = plt.figure( figsize=(10, 8))
    alt = np.arange(73, 0, -1)
    ax1 = fig.add_subplot(121)
    ax1.plot(L[0,0],alt)
    ax1.plot(L[1,0],alt)
    ax1.set_title("flx")

    ax2 = fig.add_subplot(122)
    ax2.plot(L[0,1],alt)
    ax2.plot(L[1,1],alt)
    ax2.set_title("dfdts")
    return fig

def Plot_d(L):
    alt = np.arange(72, 0, -1)
    fig, ax = plt.figure( figsize=(10, 8))
    ax.plot(L[0, 0, 1:] - L[0, 0, :72], alt)
    ax.plot(L[1, 0, 1:] - L[1, 0, :72], alt)
    ax.title('dflx')
    return fig

#########################
file1 = 'Data_net4/output0.txt'
plotfile = 'Data_net4/q0.png'
id_profile = 0
#######################

#########################
id_profile = 0
file1 = 'Data_net4/output_c2.txt'
plotfile = 'Data_net4/q2_cloud_'+str(id_profile)+'_.png'
plotfile_d = 'Data_net4/d_q2_cloud_'+str(id_profile)+'_.png'
#######################

id=-1
Profiles = []
profile = np.zeros((2,2,73))
j0=0
for i,line in enumerate(open(file1)):
    l = line.rstrip('\n').strip(' ')
    l = l.split(' ')
    if(l[0]=='Profile'):
        j0=-3
        Profiles.append(profile.copy())
        profile = np.zeros((2,3,73))
        id=-1

    if(j0>=0):
        if(len(l)==1):
            id+=1
            j=0
        else:
            if(l[0][0]=='-'):
                profile[0,id,j] = -float(l[0][1:])
            else:
                profile[0,id,j] = float(l[0])
            if(l[-1][0]=='-'):
                profile[1,id,j] = -float(l[6][1:])
            else:
                profile[1,id,j] = float(l[6])
            j+=1
    j0+=1

Profiles.append(profile)
Profiles = Profiles[1:]

Plot(Profiles[id_profile])
plt.savefig(plotfile, bbox_inches='tight')

Plot_d(Profiles[id_profile])
plt.savefig(plotfile_d, bbox_inches='tight')
