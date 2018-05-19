#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Authors: Benjamín Salomón Noyola García, Suemi Rodriguez Romo
# Plot Streamlines

import numpy as np
from StringIO import StringIO 
import matplotlib.pyplot as plt
 
# load txt file
pfile=open('T.txt','r')
pfile_1=open('vel_ux.txt','r') 
pfile_2=open('vel_uy.txt','r') 
pfile_3=open('den.txt','r')


data=pfile.read()
pfile.close()
T=np.genfromtxt(StringIO(data))

data_1=pfile_1.read()
pfile_1.close()
u=np.genfromtxt(StringIO(data_1))

data_2=pfile_2.read()
pfile_2.close()
v=np.genfromtxt(StringIO(data_2))

data_3=pfile_3.read()
pfile_3.close()
rho=np.genfromtxt(StringIO(data_3))

n, m = rho.shape
strf=np.zeros([n,m])
strf[0,0]=0.0
for j in range(m):
    rhoav=0.5*(rho[j-1,0]+rho[j,0])
    if j != 0.0: strf[j,0] = strf[j-1,0]-rhoav*0.5*(v[j-1,0]+v[j,0])
    for i in range(1,n):
        rhom=0.5*(rho[j,i]+rho[j,i-1])
        strf[j,i]=strf[j,i-1]+rhom*0.5*(u[j,i-1]+u[j,i])

strf = np.transpose(strf)
T = np.transpose(T)

strf_=np.zeros([n,m])
T_=np.zeros([n,m])
for j in range(m): 
    k=0
    for i in range(n-1,-1,-1):
        strf_[k,j] = abs(strf[i,j])
        T_[k,j] = T[i,j]

        k=k+1
xlist = np.linspace(0, n, 256)
ylist = np.linspace(0, m, 256)
X, Y = np.meshgrid(xlist, ylist)

plt.figure()
cp = plt.contour(X, Y, strf_)
plt.clabel(cp, inline=True, 
          fontsize=9)


fig = plt.figure(1) 
plt.title('Temperature and streamlines')
plt.xlabel('X')
plt.ylabel('Y')
cs1 = plt.contourf((T_), 500) # Pintamos 500 niveles con relleno
plt.colorbar()
plt.savefig("strf.jpg")
#plt.savefig("strf.eps")
plt.show()
