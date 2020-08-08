
# Autor: Benjamín Salomón Noyola García, Suemi Rodriguez Romo

import numpy as np
from io import StringIO
import matplotlib.pyplot as plt
 
#################################################
# Aqui importamos la matriz que deseamos graficar
# a partir del archivo de texto
#################################################
pfile=open('Results/T_.txt', 'r')
pfile_1=open('Results/vel_ux.txt', 'r')
pfile_2=open('Results/vel_uy.txt', 'r')
pfile_3=open('Results/den.txt', 'r')


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

##############################################
# Generación de líneas de corrente
##############################################
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
        strf_[k,j] = strf[i,j]
        T_[k,j] = strf[i,j]
        k=k+1

################################################
# Generación de gráfico de velocidades
################################################
xlist = np.linspace(0, n, 256)
ylist = np.linspace(0, m, 256)
X, Y = np.meshgrid(xlist, ylist)

plt.figure()
cp = plt.contour(X, Y, strf_)
plt.clabel(cp, inline=True, 
          fontsize=9)
plt.title('Contour Plot')

fig = plt.figure(1) 
plt.title(u'Líneas de corriente')
plt.xlabel('x')
plt.ylabel('y')
cs1 = plt.contourf((strf_), 350, cmap='coolwarm') # Pintamos 100 niveles con relleno
plt.colorbar()
plt.show()
