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
# Performed using Python-Numba-CUDA


import numpy as np
from numba import jit
from numba import cuda
import numba
import matplotlib.pyplot as plt
import d2q9_nxnyns_numba_cuda as d2q9

@jit	# just in time
def init_sol(g, w_s, T):            #initial solution 
    for i in xrange(lattice_x):
        for j in xrange(lattice_y):
        	for k in xrange(5):
        		g[i][j][k] = w_s[k]*T[i][j]   

@jit     
def prop(g):  						# streaming step
    nx, ny, ns = g.shape
    for i in xrange (ny):			# horizontal
        for j in xrange (nx-1,0,-1): 
            g[j][i][1] = g[j-1][i][1] #vector 1
        for	j in xrange (nx-1):
            g[j][i][3] = g[j+1][i][3] #vector 3
    	
    for i in xrange (ny-1): 		# vertical
        for j in xrange (nx): 
            g[j][i][2] = g[j][i+1][2] #vector 2
    	
    for i in xrange (ny-1,0,-1): 
        for j in xrange (nx):  
            g[j][i][4] = g[j][i-1][4] #vector 4          
    
@jit
def condiciones_fron(g, d_ws, d_Tb, d_Ti): # Dirichlet boundary conditions
    nx, ny, ns = g.shape
    for i in xrange(ny):
        g[0][i][1] = d_ws[1]*d_Tb[0] + d_ws[3]*d_Tb[0] - g[0][i][3] # hot
        g[nx-1][i][3] = d_ws[1]*d_Ti[0] + d_ws[3]*d_Ti[0] - g[nx-1][i][1] #cold
		
    for j in xrange(nx): 					# Periodic boundary conditions
        g[j][ny-1][2] = g[j][0][2]	# up
        g[j][0][4] = g[j][ny-1][4]	# down

if __name__ == '__main__':
    
    ##################################
    #####definition of parameters#####       
    ##################################
    rho = 1.0		# density
    Delta_alfa=10.0 # thermal diffusivity ratio
    T_i = -1.0		# low temperature
    T_b = 1.0 		# high temperature
    T_m = 0.0		# melting temperature
    alpha_s = 0.002	# thermal diffusivity of solid
    alpha_l = Delta_alfa*alpha_s # thermal diffusivity of liquid
    poros = 1.0 	# porosity
    sigma = 1.0 	# thermal capacity ratio
    Cpl=Cps = 1.0 	# specific heat
    w_test = -2.0 	# constant of D2Q5 MRT-LB model
    k_s = alpha_s*rho*Cps  # solid thermal conductivity
    k_l = alpha_l*rho*Cpl  # liquid thermal conductivity
    St = 1.0    	# Stefan number
    F_0 = 0.01  	# Fourier number
    H = 200.0   	# characteristic leght
    La = Cpl*(T_b-T_m)/St 	# latent heat
    H_l = Cpl*0.02 + 1.0*La # Enthalpy of liquid  0.04/2 = 0.02,  T_m=0 -> Tl=0.02   
    H_s = Cps*(-0.02) + 0.0*La # Enthalpy of solid  Ts=-0.02
    t = (F_0*H**2)/alpha_s  # Time
    delta_x = delta_t=1.0   # time and space step
    c = delta_x/delta_t     # lattice speed
    c_st = np.sqrt(1.0/5.0) #sound speed of the D2Q9 model
    pasos = int(t)  
    lattice_x = 256 # lattices in x direction; edit this line to change the size
    lattice_y = 8   # lattices in y direction; edit this line to change the size 
    pasos=300000	   # Edit this line to change the time steps and then change the time steps in analytic_vs_num_plot.py **
    print "Steps =", pasos # Steeps
    ##################################
    ############### arrays ###########       
    ##################################   
    T    = np.ones([lattice_x,lattice_y]) # Temperature is saved in a matrix
    g = np.zeros([lattice_x,lattice_y,5]) # distribution function is saved in a tensor order 3
    h_copia_g = np.zeros([lattice_x,lattice_y,5])    
    f_l  = np.zeros([lattice_x,lattice_y])# liquid fraction is saved in a matrix
    w_s = np.zeros([5]) 				  # weight coefficients
    w_s[0] = (1.0-w_test)/5.0
    for i in xrange(1,5,1):
        w_s[i] = (4.0+w_test)/20.    
 
    ###################################
    ########paramethers of CUDA########       
    ###################################
    nx, ny, ns  = lattice_x, lattice_y, 5	
    threads = 256,4 #1024#512
    blocks = (nx/threads[0]+(0!=nx%threads[0]),
              ny/threads[1]+(0!=ny%threads[1]) )
#    print threads

    ###################################
    ####### The solution begins #######
    ###################################

    #### Initial conditions  ####
    T=T_i*T
    T[0,:], T[lattice_x-1,:] = T_b, T_i # Initial temperature
    f_l[0,:], f_l[lattice_x-1,:] = 1.0, 0.0 # Initial liquid fraction
    init_sol(g, w_s, T) 				# Initial distribution functions
    
    ### numerical arrays sends to GPU ###
    T_b, T_i, D_al = np.array([float(T_b)]), np.array([float(T_i)]), np.array([float(Delta_alfa)])    
    d_g = cuda.to_device(g)
    D_al = cuda.to_device(D_al)   
    d_fl = cuda.to_device(f_l) 
        
    import time
    tiempo_cuda = time.time() # Messures the time at the beginning 
    
    for ii in xrange(pasos): # main loop
        d2q9.one_time_step[blocks, threads](d_g, d_fl, D_al)
        d_g.to_host()		
        d_fl.to_host()
        prop(g)
        condiciones_fron(g, w_s, T_b, T_i)#
        d_g = cuda.to_device(g)
        d_fl = cuda.to_device(f_l)

    d_g.to_host()
    d_fl.to_host()
    T = g.sum(axis=2)    
    
    t = time.time() - tiempo_cuda
    Perfo=[t, lattice_x*lattice_y*pasos/t/1e6]  # simulation time, MLUPS    
    T=np.transpose(T) 
    f_l=np.transpose(f_l)    
    print"time, MLUPS/2 = ", t, nx*ny*pasos/t/1e6 # print time and MLUPS
    print "\n Temperature: \n",T
    print "\n Liquid fraction \n",f_l    

    np.savetxt('Performance_256X8_300mil.txt',Perfo)   # Save performance
    np.savetxt('T_256X8_300mil.txt', T,fmt='%.6f'  )   # Save temperature
    np.savetxt('f_l_256X8_300mil.txt', f_l,fmt='%.3f') # Save liquid fraction
