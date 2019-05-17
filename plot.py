import numpy as np
import matplotlib.pyplot as plt

# Plot output of the FOrtran model

def Plot(L):
    fig = plt.figure( figsize=(10, 8))
    alt = np.arange(73)
    ax1 = fig.add_subplot(221)
    plt.plot(L[0,0],alt)
    plt.plot(L[1,0],alt)
    plt.title("flxu")

    ax2 = fig.add_subplot(222)
    plt.plot(L[0,1],alt)
    plt.plot(L[1,1],alt)
    plt.title("flxd")

    ax3 = fig.add_subplot(223)
    plt.plot(L[0,2],alt)
    plt.plot(L[1,2],alt)
    plt.title("dfdts")

    ax4 = fig.add_subplot(224)
    plt.plot(L[0,0]+L[0,1],alt)
    plt.plot(L[1,0]+L[1,1],alt)
    plt.title("flx");

#########################
file1 = 'output.txt'
plotfile = 'q.png'
id_profile = 0
#######################

id=-1
Profiles = []
profile = np.zeros((2,3,73))
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
                profile[1,id,j] = -float(l[-1][1:])
            else:
                profile[1,id,j] = float(l[-1])
            j+=1
    j0+=1

Profiles.append(profile)
Profiles = Profiles[1:]

Plot(Profiles[id_profile])
plt.savefig(plotfile, bbox_inches='tight')
