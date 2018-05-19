#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Authors: Benjamín S. Noyola García, Suemi Rogriguez Romo
# 
# Numerical Solution 1D performed by MRT-LB, Lattice of 256*8, 
# the phase change of a bar is modeled in 1D section 4.1 of 
# paper: Qing, Liu, Ya-Ling He*. Double multiple-relaxation-time 
# lattice Boltzmann Model for solid-liquid phase change with natural 
# convection in porous media, Physica A. 438(2015) 94-106.
#
# Performed with python-numba

from numpy import *
import matplotlib.pyplot as plt
from numba import jit

#...Parameters...#
rho = 1.0   # density
Delta_alfa=10.0 # thermal diffusivity ratio
T_i =-1.0   # low temperature
T_b = 1.0   # high temperature
T_m = 0.0   # melting temperature
alpha_s = 0.002 # thermal diffusivity of solid
alpha_l = Delta_alfa*alpha_s  # thermal diffusivity of liquid
poros = 1.0 # porosity
sigma = 1.0 # thermal capacity ratio
Cpl=Cps = 1.0 # specific heat
w_test = -2.0 # constant of D2Q5 MRT-LB model
k_s = alpha_s*rho*Cps # solid thermal conductivity 
k_l = alpha_l*rho*Cpl # liquid thermal conductivity
St = 1.0    # Stefan number
F_0 = 0.01  # Fourier number
H = 256.0   # characteristic leght
La = Cpl*(T_b-T_m)/St   # latent heat
H_l = Cpl*0.02 + 1.0*La # Enthalpy of liquid  0.04/2 = 0.02,  T_m=0 -> Tl=0.02   
H_s = Cps*(-0.02) + 0.0*La # Enthalpy of solid  Ts=-0.02
t = (F_0*H**2)/alpha_s  # time 
delta_x = delta_t=1.0   # time and space step
c = delta_x/delta_t     # lattice speed
c_st = sqrt(1.0/5.0)    #sound speed of the D2Q9 model
lattice_x = 256    # lattices in x direction; edit this line to change the size
lattice_y = 8       # lattices in y direction; edit this line to change the size 
pasos = 300000      # Edit this line to change the time steps and then change the time steps in analytic_vs_num_plot.py **
print"steps: ", pasos

#Lists:
T    = ones([lattice_y,lattice_x]) # Temperature is saved in a matrix
g = zeros([5,lattice_y,lattice_x]) # distribution function is saved in a tensor order 3
g_eq = zeros([5,lattice_y,lattice_x]) # equilibrium distribution function is saved in a tensor order 3
n = zeros([5,lattice_y,lattice_x]) # distribution function in moment space
n_eq = zeros([5,lattice_y,lattice_x]) # Equilibrium distribution function in moment space
n_res1 = zeros([5]) # distribution function after collision
n_res2 = zeros([5]) # distribution function after collision 
H_k  = zeros([lattice_y,lattice_x])	# entalpy is saved in a matrix 
f_l  = zeros([lattice_y,lattice_x])	# liquid fraction is saved in a matrix
t_relax_T_ad = zeros([lattice_y,lattice_x])	# tiempo de relajación adimencional del campo de temperatura
relax_o = zeros([lattice_y,lattice_x])  
alpha_e = zeros([lattice_y,lattice_x])	# thermal diffusivity 
tau_t = zeros([lattice_y,lattice_x])	# relaxation parameters 
k_e = zeros([lattice_y,lattice_x])		# thermal conductivity
w_s = zeros([5]) 		# weight coefficients 
s_surce = zeros([5]) 	# Surce 
N=array([[1.0, 1.0, 1.0, 1.0,1.0],[0.0, 1.0, 0.0, -1.0, 0.0],[0.0, 0.0, 1.0, 0.0, -1.0],[-4.0, 1.0, 1.0, 1.0,1.0],[0.0, 1.0, -1.0, 1.0, -1.0]]) # matrix transformation between moment space and velocity space
N_inv= linalg.inv(N)    # inverse matrix
#print N, N_inv 

for i in range(5):      # wheight coeficients for the temperature field:
	if i == 0:
		w_s[i] = (1-w_test)/5.
	else:
		w_s[i] = (4+w_test)/20.


T=T_i*T  # initial conditions
for i in range (lattice_y):
	T[i][0]=T_b            # high temperature in boundary    
	T[i][lattice_x-1] = T_i# low temperature in boundary 
	f_l[i][0]=1.0          # liquid fraction
	f_l[i][lattice_x-1] =0.0 
f_2l=copy(f_l)

for i in range(lattice_y): # cálculo de entalpia en cada lattice
		for j in range(lattice_x):
			H_k[i][j] = Cps*T[i][j] + f_l[i][j]*La


for i in range(lattice_y): # initial distribution function 
	for j in range(lattice_x):
		for k in range(5):
			g[k][i][j] = w_s[k]*T[i][j] 

@jit  # compile just in time
def MRT_LB(pasos,g):
	for kk in range(pasos):
		
		T = g.sum(axis=0) # Temperature calculation
	
		f_2l=copy(f_l) # copy the liquid fraction 
		for i in xrange(lattice_y): # Enthalpy calculation
			for j in xrange(lattice_x):
				H_k[i][j] = Cps*T[i][j] + f_l[i][j]*La
				if (H_k[i][j] <= H_s):
					f_l[i][j]=0.0
				elif (H_k[i][j] > H_s and H_k[i][j] < H_l):
					f_l[i][j] = (H_k[i][j] - H_s)/(H_l - H_s)
				else: 
					f_l[i][j]=1.0
		
	
		n = tensordot(N, g, axes=([1],[0])) # Transformation from velocity space to moment space
	
	
	# (1) collision......................................... 	
		for i in xrange(lattice_y):
			for j in xrange(lattice_x):
				alpha_e[i][j] = alpha_l*f_l[i][j] + alpha_s*(1.0-f_l[i][j]) 
				tau_t[i][j] = 0.5 + alpha_e[i][j]/(sigma*c_st**2.0*delta_t)
	
		for i in xrange(lattice_y):  
			for j in xrange(lattice_x):
				relax_o = array([[1.0, 0.0, 0.0, 0.0, 0.0],[0.0, 1.0/(tau_t[i][j]), 0.0, 0.0, 0.0],[0.0, 0.0, 1.0/(tau_t[i][j]), 0.0, 0.0],[0.0, 0.0, 0.0, 1.5, 0.0],[0.0, 0.0, 0.0, 0.0, 1.5]])
				s_surce = array([-((poros*La)/(sigma*Cpl))*(f_l[i][j]-f_2l[i][j])/1.0 , 0.0 , 0.0 , -w_test*((poros*La)/(sigma*Cpl))*(f_l[i][j]-f_2l[i][j])/1.0 , 0.0])
				n_eq = array([T[i][j] , 0.0 , 0.0 , w_test*T[i][j] , 0.0]) # equilibrium distribution function
				for k in xrange(5):
					n_res1[k] = n[k][i][j] - n_eq[k]
				n_res2=dot(relax_o,n_res1)
				for k in xrange(5):
					n[k][i][j] = n[k][i][j] - n_res2[k] + s_surce[k]  # collision step.
	
		g = tensordot(N_inv, n, axes=([1],[0])) #transformation from moment space to velocity space
				
		# Streaming step...................................................
		for i in xrange (lattice_y):
			for j in xrange (lattice_x-1,0,-1): # horizontal
				g[1][i][j] = g[1][i][j-1] #vector 1
			for	j in xrange (lattice_x-1):
				g[3][i][j] = g[3][i][j+1] #vector 3
	
		for i in xrange (lattice_y-1): # Vertical
			for j in xrange (lattice_x): 
				g[2][i][j] = g[2][i+1][j] #vector 2
	
		for i in xrange (lattice_y-1,0,-1): 
			for j in xrange (lattice_x):			
				g[4][i][j] = g[4][i-1][j] #vector 4
	
		# boundadry condition.......................................
		for i in xrange(lattice_y): #Dirichlet boundary condition (vertical)
			g[1][i][0] = w_s[1]*T_b + w_s[3]*T_b - g[3][i][0]
			g[3][i][lattice_x-1] = w_s[1]*T_i + w_s[3]*T_i - g[1][i][lattice_x-1]
		
		for j in xrange(lattice_x): # Periodic boundary conditions (horizontal)
			g[2][lattice_y-1][j] = g[2][0][j]
			g[4][0][j] = g[4][lattice_y-1][j]
			
	return g


import time
tiempo_numba = time.time() # Measure computing time (beginning)
gg=MRT_LB(pasos,g)
T = gg.sum(axis=0)
t = time.time() - tiempo_numba # Measure computing time
print"\n", t, lattice_x*lattice_y*pasos/t/1e6 # Report MLUPS   
print "T= ",T
Perfo=[t, lattice_x*lattice_y*pasos/t/1e6]
import numpy as np # solo para guargar los resultados en archivos de texto
np.savetxt('Perfo_256X8_300mil.txt',Perfo) # Save performance
np.savetxt('T_256X8_300mil.txt', T,fmt='%.6f')# Save temperature
np.savetxt('f_l_256X8_300mil.txt', f_l,fmt='%.3f')# Save liquid fraction

