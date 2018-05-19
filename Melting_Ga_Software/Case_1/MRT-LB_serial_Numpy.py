#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Autors: Benjamín S. Noyola García, Suemi Rogriguez Romo
# 
# Numerical Solution 1D performed by MRT-LB, Lattice of 254*8, 
# the phase change of a bar is modeled in 1D section 4.1 of 
# paper: Qing, Liu, Ya-Ling He*. Double multiple-relaxation-time 
# lattice Boltzmann Model for solid-liquid phase change with natural 
# convection in porous media, Physica A. 438(2015) 94-106.
#
# Performed with python-numpy

from numpy import *
import matplotlib.pyplot as plt
import time

#...Parameters...#
rho = 1.0
Delta_alfa=10.0 # thermal diffusivity ratio
T_i =-1.0 # low temperature
T_b = 1.0 # high temperature
T_m = 0.0 # melting temperature
alpha_s = 0.002 # thermal diffusivity of solid
alpha_l = Delta_alfa*alpha_s # thermal diffusivity of liquid
poros = 1.0     # porosity
sigma = 1.0     # thermal capacity ratio
Cpl=Cps = 1.0   # specific heat
w_test = -2.0   # constant of D2Q5 MRT-LB model
k_s = alpha_s*rho*Cps # solid thermal conductivity
k_l = alpha_l*rho*Cpl # liquid thermal conductivity
St = 1.0        # Stefan number
F_0 = 0.01      # Fourier number
H = 256.0       # characteristic leght
La = Cpl*(T_b-T_m)/St   # latent heat
H_l = Cpl*0.02 + 1.0*La # Enthalpy of liquid  0.04/2 = 0.02,  T_m=0 -> Tl=0.02   
H_s = Cps*(-0.02) + 0.0*La # Enthalpy of solid  Ts=-0.02
t = (F_0*H**2)/alpha_s  # time step  
delta_x = delta_t=1.0   # time and space step
c = delta_x/delta_t # velocidad de lattice
c_st = sqrt(1.0/5.0)#sound speed of the D2Q9 model
pasos=300000    # Edit this line to change the time steps and then change the time steps in analytic_vs_num_plot.py **
lattice_x = 256 # lattices in x direction; edit this line to change the size
lattice_y = 8   # lattices in y direction; edit this line to change the size 
print"steps: ", pasos

#arrays:
T    = ones([lattice_y,lattice_x]) # Temperature is saved in a matrix
g = zeros([5,lattice_y,lattice_x]) # distribution function is saved in a tensor order 3
g_eq = zeros([5,lattice_y,lattice_x]) # equilibrium distribution function is saved in a tensor order 3
n = zeros([5,lattice_y,lattice_x]) # distribution function in moment space
n_eq = zeros([5,lattice_y,lattice_x]) # Equilibrium distribution function in moment space
n_res1 = zeros([5]) # distribution function after collision
n_res2 = zeros([5]) # distribution function after collision
H_k  = zeros([lattice_y,lattice_x])			# entalpy is saved in a matrix 
f_l  = zeros([lattice_y,lattice_x])			# liquid fraction is saved in a matrix
t_relax_T_ad = zeros([lattice_y,lattice_x])	# tiempo de relajación adimencional del campo de temperatura
relax_o = zeros([lattice_y,lattice_x])
alpha_e = zeros([lattice_y,lattice_x])		# thermal diffusivity
tau_t = zeros([lattice_y,lattice_x])			# relaxation parameters
k_e = zeros([lattice_y,lattice_x])			# thermal conductivity
w_s = zeros([5]) 		# weight coefficients 
s_surce = zeros([5]) 	# Surce 
N=array([[1.0, 1.0, 1.0, 1.0,1.0],[0.0, 1.0, 0.0, -1.0, 0.0],[0.0, 0.0, 1.0, 0.0, -1.0],[-4.0, 1.0, 1.0, 1.0,1.0],[0.0, 1.0, -1.0, 1.0, -1.0]])
N_inv= linalg.inv(N)
#print N, N_inv

# weights:
for i in range(5):
	if i == 0:
		w_s[i] = (1-w_test)/5.
	else:
		w_s[i] = (4+w_test)/20.
	
# initial conditions
T=T_i*T
for i in range (lattice_y):
	T[i][0]=T_b
	T[i][lattice_x-1] = T_i
	f_l[i][0]=1.0
	f_l[i][lattice_x-1] =0.0
f_2l=copy(f_l)

for i in range(lattice_y): # calculation of enthalpy at the beginning
		for j in range(lattice_x):
			H_k[i][j] = Cps*T[i][j] + f_l[i][j]*La

# initial distribution function
for i in range(lattice_y):
	for j in range(lattice_x):
		for k in range(5):
			g[k][i][j] = w_s[k]*T[i][j]
#print "g_initial: ",g

tiempo_inicio = time.time()
for kk in range(pasos):

	
	T = g.sum(axis=0) #caltulation of temperature

	f_2l=copy(f_l) # get a copy of liquid fraction
	for i in xrange(lattice_y): # calculation of enthalpy
		for j in xrange(lattice_x):
			H_k[i][j] = Cps*T[i][j] + f_l[i][j]*La
			if (H_k[i][j] <= H_s):
				f_l[i][j]=0.0
			elif (H_k[i][j] > H_s and H_k[i][j] < H_l):
				f_l[i][j] = (H_k[i][j] - H_s)/(H_l - H_s)
			else: 
				f_l[i][j]=1.0
	
# (3) transformación de la función de distribución espacio de vel a espacio momentos
	n = tensordot(N, g, axes=([1],[0])) 


# (4) colision step
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
				n[k][i][j] = n[k][i][j] - n_res2[k] + s_surce[k]  # main colisión

	# (5) ontain temperature and liquid fraction
	g = tensordot(N_inv, n, axes=([1],[0])) # linear transformation frum moment space to velocity space
	
	
	# Propagation step
	for i in xrange (lattice_y):
		for j in xrange (lattice_x-1,0,-1): # propagate horizontal distribution functions (df)
			g[1][i][j] = g[1][i][j-1] # df 1
		for	j in xrange (lattice_x-1):
			g[3][i][j] = g[3][i][j+1] # df 3

	for i in xrange (lattice_y-1): # propagate vertical distribution functions 
		for j in xrange (lattice_x): 
			g[2][i][j] = g[2][i+1][j] # df 2

	for i in xrange (lattice_y-1,0,-1): 
		for j in xrange (lattice_x):			
			g[4][i][j] = g[4][i-1][j] # df 4

	# () boundary conditions
	for i in xrange(lattice_y):
		g[1][i][0] = w_s[1]*T_b + w_s[3]*T_b - g[3][i][0]
		g[3][i][lattice_x-1] = w_s[1]*T_i + w_s[3]*T_i - g[1][i][lattice_x-1]
	
	for j in xrange(lattice_x): #condiciones de frontera periodicas horizontales
		g[2][lattice_y-1][j] = g[2][0][j]
		g[4][0][j] = g[4][lattice_y-1][j]
		

tiempo_fin = time.time()
t = tiempo_fin - tiempo_inicio
#print"\n", t, lattice_x*lattice_y*pasos/t/1e6
Perfo=[t, lattice_x*lattice_y*pasos/t/1e6]  # simulation time, MLUPS
print "time:",t
#print "N: ",N
#print "N_inv: ",N_inv
#print N.shape
#print g.shape
T = g.sum(axis=0)
print "Temperature= ",T
savetxt('Performance_256X8_300mil.txt',Perfo) # Save performance
savetxt('T_256X8_300mil.txt', T,fmt='%.6f'  ) # Save temperature
savetxt('f_l_256X8_300mil.txt', f_l,fmt='%.3f') # Save liquid fraction
