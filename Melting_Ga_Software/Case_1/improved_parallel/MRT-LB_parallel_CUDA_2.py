#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Authors: Benjamín Salomón Noyola García, Suemi Rodriguez Romo

# Numerical Solution 1D performed by MRT-LB, Lattice of 256*8, 
# the phase change of a bar is modeled in 1D section 4.1 of 
# paper: Qing, Liu, Ya-Ling He*. Double multiple-relaxation-time 
# lattice Boltzmann Model for solid-liquid phase change with natural 
# convection in porous media, Physica A. 438(2015) 94-106.
#
# Performed using Python-Numba-CUDA

import numpy as np
from numba import jit, cuda

import matplotlib.pyplot as plt
import MRT_LB_shared as d2q5

@jit
def init_sol(g, w_s, T):
    for i in xrange(lattice_x):
        	for j in xrange(lattice_y):
        		for k in xrange(5):
        			g[i][j][k] = w_s[k]*T[i][j]   

@jit
def condiciones_fron(g, d_ws, d_Tb, d_Ti):
    nx, ny, ns = g.shape
    for i in xrange(ny): # Boundary conditions in east and west 
        g[0][i][1] = d_ws[1]*d_Tb[0] + d_ws[3]*d_Tb[0] - g[0][i][3]
        g[nx-1][i][3] = d_ws[1]*d_Ti[0] + d_ws[3]*d_Ti[0] - g[nx-1][i][1]
		
    for j in xrange(nx): # Boundary conditions in North and south
        g[j][ny-1][2] = g[j][0][2]
        g[j][0][4] = g[j][ny-1][4]
        

if __name__ == '__main__':
    
    ##################################
    ##### Parameters #####       
    ##################################
    rho = 1.0			   # density
    Delta_alfa=10.0 	       # thermal diffusivity ratio
    T_i = -1.0   		   # low temperature
    T_b = 1.0 			   # high temperature
    T_m = 0.0			   # melting temperature
    alpha_s = 0.002 	       # thermal diffusivity of solid
    alpha_l = Delta_alfa*alpha_s  # thermal diffusivity of liquid
    poros = 1.0 		      # porosity
    sigma = 1.0 		      # thermal capacity ratio
    Cpl=Cps = 1.0 		  # specific heat
    w_test = -2.0 		  # constant of D2Q5 MRT-LB model
    k_s = alpha_s*rho*Cps  # solid thermal conductivity 
    k_l = alpha_l*rho*Cpl  # liquid thermal conductivity
    St = 1.0    # Stefan number
    F_0 = 0.01  # Fourier number
    H = 256.0   # characteristic leght, this is redefined later.
    La = Cpl*(T_b-T_m)/St   # latent heat
    H_l = Cpl*0.02 + 1.0*La # Enthalpy of liquid
    H_s = Cps*(-0.02) + 0.0*La # Enthalpy of solid
    t = (F_0*H**2)/alpha_s  #Tieme
    delta_x = delta_t=1.0   #time and space step
    c = delta_x/delta_t     # lattice speed
    c_st = np.sqrt(1.0/5.0) #sound speed of the D2Q5 model
    lattice_x = 256 	       # lattices in x direction; edit this line to change the size
    lattice_y = 8 		   # lattices in y direction; edit this line to change the size 
    pasos=300000 		   # Edit this line to change the number of steps
    print "Steps =", pasos    
    
    ##################################
    ############# Arrays #############       
    ##################################   
    T = np.ones([lattice_x,lattice_y])          # temperature is saved in a matrix
    g = np.zeros([lattice_x,lattice_y,5])       # distribution function is saved in a tensor order 3
    h_copia_g = np.zeros([lattice_x,lattice_y,5]) # a copy of g   
    f_l  = np.zeros([lattice_x,lattice_y])	    # liquid fraction is saved in a matrix
    w_s = np.zeros([5]) 		                 # weight coefficients 
    w_s[0] = (1.0-w_test)/5.0
    for i in xrange(1,5,1):
        w_s[i] = (4.0+w_test)/20.    
 
    ##################################
    ######## CUDA parameters ########       
    ##################################
    nx, ny, ns  = lattice_x, lattice_y, 5
    threads = 256,4 #1024#512@@@@@@@@@@@@@@@@@@@@ You can change CUDA parameter @@@@@@@@@@@@@
    blocks = (nx/threads[0]+(0!=nx%threads[0]),
              ny/threads[1]+(0!=ny%threads[1]))

    # print threads
    ##################################
    ####### solution begins #######
    ##################################
#    
    #### initial conditions  ####
    T=T_i*T
    T[0,:], T[lattice_x-1,:] = T_b, T_i     # initial temperature
    f_l[0,:], f_l[lattice_x-1,:] = 1.0, 0.0 # initial liquid fraction
    init_sol(g, w_s, T)                     # initial solucion (g)
    
    ### Main CUDA loop  ###
    T_b, T_i, D_al = np.array([float(T_b)]), np.array([float(T_i)]), np.array([float(Delta_alfa)])        
    d_g = cuda.to_device(g)  # send g from host to device
    copia_g = cuda.to_device(h_copia_g) # send a copy of g from host to device
    d_ws = cuda.to_device(w_s)	# send w_s from host to device
    D_al = cuda.to_device(D_al)	# send thermal diffusivity ratio from host to device
    d_fl = cuda.to_device(f_l) 	# send liquid fraction from host to device
    
    import time
    tiempo_cuda = time.time()   # Measure initial time, before the main CUDA loop begins
    for ii in xrange(pasos):
        d2q5.collision_local[blocks, threads](d_g, d_fl, D_al) # colision step in local memory
        d2q5.propagacion[blocks, threads](d_g,copia_g)         # streaming step in shared memory
        
#        d_g.to_host()                      # uncoment this line for 1024X8, 2048 size and comment 115 
#        condiciones_fron(g, w_s, T_b, T_i) # uncoment this line for 1024X8, 2048 size and comment 115
#        d_g = cuda.to_device(g)            # uncoment this line for 1024X8, 2048 size and comment 115

        d2q5.condiciones_frontera[blocks, threads](d_g, d_ws) #comentar esta línea y descomentar 113-115
    d_g.to_host()       # copy g to host 
    d_fl.to_host()      # copy liquid fraction (f_l) to host 
    T = g.sum(axis=2)   # calculate temperature from the distribution function 
    t = time.time() - tiempo_cuda   # Measure final time, after the main CUDA loop ends
    print"time, MLUPS/2 = ", t, nx*ny*pasos/t/1e6 # print time and MLUPS
    Perfo=[t, nx*ny*pasos/t/1e6]  # simulation time, MLUPS
    T=np.transpose(T) 
    f_l=np.transpose(f_l)    
    print "\n Temperature: \n",T    # Print temperature as a matrix, every cell corresponds to a lattice
    print "\n Liquid fraction \n",f_l  # Print Liquid fraction as a matrix, every cell corresponds to a lattice
    np.savetxt('Perfo_256X8_300mil.txt',Perfo) # Save in txt the performance
    np.savetxt('T_256X8_300mil.txt', T,fmt='%.6f') # Save in txt the temperature
    np.savetxt('f_l_256X8_300mil.txt', f_l,fmt='%.3f') # Save in txt liquid fraction 