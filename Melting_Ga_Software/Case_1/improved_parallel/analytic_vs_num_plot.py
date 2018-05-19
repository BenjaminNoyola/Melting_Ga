#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Autor: Benjamín S. Noyola García
# plot analytic solution Vs Numric solution

from numpy import *
import matplotlib.pyplot as plt
from scipy.special import erfc
from scipy.special import erf
from scipy.optimize import brentq
#.................. Load txt file of temperature ..................#
matriz=loadtxt("T_256X8_300mil.txt")
size=256 ######## Edit this line to change the mesh size ###########
ydata=zeros([size])
for i in range(size):
	ydata[i]=matriz[5][i]
xdata = linspace(0, size-1,size)

#...................... Analytic solution ...........................#
#...Parameters...#
Delta_alfa=10.0 
T_i =-1.0
T_b = 1.0
T_m = 0.0
alpha_s = 0.002
alpha_l = Delta_alfa*alpha_s
poros = 1.0
sigma = 1.0
Cpl=Cps=1.0
rho = 1.0
k_s=alpha_s*rho*Cps
k_l=alpha_l*rho*Cpl
F_0=0.01
H=256.0
t=(F_0*H**2)/alpha_s
t=300000  # Edit this line
print "\nAnalytic time steps: ",t
#...Solution of a no linear equation #
def f(n):
	La=Cpl*(T_b-T_m)
	return (exp(-n**2))/(erf(n)) + (k_s/k_l)*sqrt(alpha_l/alpha_s)*((T_i-T_m)/(T_b-T_m))* \
	(exp(-n**2*(alpha_l/alpha_s))/erfc(n*sqrt(alpha_l/alpha_s))) - ((n*La*sqrt(pi))/(Cpl*(T_b-T_m)))

n=brentq(f, 0.0, 5.0) #  brentq function
print n
xm = 2.0*n*sqrt(alpha_l*t)

print "xm = ",xm,"\n"

x = linspace(0, size-1, size)
dominio_1 = linspace(0,1,size)
dominio_2 = linspace(0,1,size)
T = zeros(size)
tetha   = zeros(size)

for i in range (size):
	if i <= (size/size)*(xm):
		T[i]=T_b-((T_b-T_m)*erf(x[i]/(2.*sqrt(alpha_l*t))))/erf(n)
		tetha[i] = (T[i]-T_b)/(T_b-T_m)
	else:
		T[i]=T_i+((T_m-T_i)*erfc(x[i]/(2.*sqrt(alpha_s*t))))/erfc(n*sqrt(alpha_l/alpha_s))
		tetha[i] = (T[i]-T_b)/(T_b-T_m)


# .............Plot of analytic vs numeric solution................#
analitica,=plt.plot(dominio_1, T, 'xr',lw=3, label="Analytic")
numerica,=plt.plot(dominio_2 , ydata,'*',lw=1, label="MRT-LB CUDA")

plt.title('Analytic vs numeric solutions')
plt.grid(True)
plt.grid(color='g', alpha=0.5, linestyle='solid', linewidth=1.0) # intentar
plt.ylabel('T')
plt.text(0.4, 0.0, r"$\alpha_l / \alpha_s = 10.0$", fontsize=30, color="blue")
plt.text(0.4, 0.2, r" t=300000", fontsize=30, color="blue")
plt.ylim(-1.2, 1)
plt.xlabel('x/H')
plt.legend()
plt.show()

