#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Authors: Benjamín S. Noyola García, Suemi Rodriguez Romo

"""Numerical solution of Solid-liquid phase change simulation of a material (Ga) 
immersed in a porous media, performed by Numba-CUDA. This simulation uses MRLBM, 
with a D2Q5 stencil, for the heat transfer in Ga and a D2Q9 stencil for the momentum 
transfers in the liquid Ga. The last one takes into account  natural convection.
This software is initially inspired in the article: Qing, Liu, Ya-Ling He*. 
Double multiple-relaxation-time lattice Boltzmann Model for solid-liquid phase 
change with natural convection in porous media, Physica A. 438(2015) 94-106.
This program performs the simulation with homogeneous porosity""" 


import numpy as np
from numpy import linalg as LA
from numba import jit,cuda
import numba, time
import matplotlib.pyplot as plt
import modulos as mod   
        
if __name__ == '__main__':
    
# 1.-................... Parameters ..............................
    lattice_x , lattice_y = 256, 256 # Lattices size
    Ra = 8.409e5 	# Rayleight number
    Da = 1.37e-5    	# Darcy number
    poros = 0.385	# Porosity
    J = 1.0			# viscosity ratio
    Pr = 0.0208 	     # Prandtl number
    Lambda = 0.2719	# efective termal difusivity ratitio, respect to the liquid difusivity
    T_h = 45.0		# hot temperture
    T_c = 20.0		# cold temperature
    Cpl=Cps = 1.0	# specific heat
    T_m = 29.78		# meltilg temperature
    T_i = T_o = 20.0  # initial temperature 
    H = float(lattice_y)	# characteristic lenght
    Fo = 1.829        # Fourier number
    St = 0.1241		# Stephan number
    Ma = 0.1 		# Mach number
    sigma_l = 0.86    # termal capacity ratio of liquid
    sigma_s = 0.8352  # termal capacity ratio of solid
    La = Cpl*(T_h-T_c)/St
    rho_o = 1.0 	     # initial density
    w_test = -2.0	
    delta_x = delta_y = delta_t=1.0 # space and time step
    c = delta_x/delta_t
    c_st = np.sqrt(1.0/5.0)
    c_s = c/np.sqrt(3.0)
    tau_v = 0.5 
    K = Da*H**2         # Permeability
    H_l = Cpl*(T_m+0.5) + 1.0*La # liquid enthalpy  0.04/2 = 0.02,  T_m=0 -> Tl=0.02   
    H_s = Cps*(T_m-0.5) + 0.0*La # solid enthalpy  Ts=-0.02
    t=int((Fo*H)/0.08731)
    
# 2.- ...............arrays of temperature field..............................
    T = np.ones([lattice_x, lattice_y])     # temperature
    g = np.zeros([lattice_x, lattice_y, 5]) # distribution function
    g_eq = np.zeros([lattice_x, lattice_y, 5]) # equilibrium distribution function
    H_k  = np.zeros([lattice_x,lattice_y])	# enthalpy 
    f_l  = np.zeros([lattice_x,lattice_y])	# liquid fraction
    w_s = np.zeros([5]) 		             # weights
    
    # velocity field arrays:
    den = np.ones([lattice_x,lattice_y])  # density
    f = np.zeros([lattice_x,lattice_y,9]) # distribution function
    f_eq = np.zeros([lattice_x,lattice_y,9]) # equilibrium distribution function
    w = np.zeros([9]) 		                # weights
    s = np.array([1.0, 1.1, 1.1, 1.0, 1.2, 1.0, 1.2, 1.0/tau_v, 1.0/tau_v]) 
    S = np.zeros([9])
    relax_f = np.zeros([9,9])
    np.fill_diagonal(relax_f,s) 
    vel_vx = np.zeros([lattice_x,lattice_y]) 
    vel_vy = np.zeros([lattice_x,lattice_y]) 
    vel_ux = np.zeros([lattice_x,lattice_y]) 
    vel_uy = np.zeros([lattice_x,lattice_y]) 
    Fx = np.zeros([lattice_x,lattice_y]) 
    Fy = np.zeros([lattice_x,lattice_y]) 
    
    # weights(temperature field):
    for k in range(5):
        if k == 0:
            w_s[k] = (1.0-w_test)/5.0
        else:
            w_s[k] = (4.0+w_test)/20.0
    
    # weight(velocity field):
    for k in range(9):
        if k == 0:
            w[k] = 4./9.
        elif (k >= 1 and k <= 4):
            w[k] = 1./9.
        else:
            w[k] = 1./36.
    
# 3.- ............... Initial conditions.......................
    # temperature field:
    T=T_i*T
    for j in range (lattice_y):
        T[0,j]=T_h
        T[lattice_x-1,j] = T_c
        f_l[0,j]=1.0
        f_l[lattice_x-1,j] =0.0
    f_2l = np.copy(f_l)
    
    # Enthalpy calculation (temperature field)
    for i in xrange(lattice_x):
        for j in xrange(lattice_y):
            H_k[i,j] = Cps*T[i,j] + f_l[i,j]*La
    
#  4.- ........... initial distribution function ..................
    for i in range(lattice_x):
        for j in range(lattice_y):
            for k in range(5):
                g_eq[i,j,k] = w_s[k]*T[i,j]  # temperature field
    g=np.copy(g_eq)
    g2=np.copy(g_eq)    

    for i in range(lattice_x):
        for j in range(lattice_y):
            for k in xrange(9):
                f_eq[i,j,k] = w[k]*den[i,j] # velocity field
    f=np.copy(f_eq)
    f2=np.copy(f_eq)
    ##################################
    ########parámetros de CUDA########       
    ##################################
    threads = 256,1 #1024#512
    blocks = (lattice_x/threads[0]+(0!=lattice_x%threads[0]),
              lattice_y/threads[1]+(0!=lattice_y%threads[1]) )
#    print threads
#5.-  comienza ciclo, envia calculo a CUDA ###
    d_f = cuda.to_device(f)
    d_f2 = cuda.to_device(f2)    
    d_g = cuda.to_device(g)
    d_g2 = cuda.to_device(g2)    
    d_vel_ux = cuda.to_device(vel_ux)
    d_vel_uy = cuda.to_device(vel_uy)
    d_Fx = cuda.to_device(Fx)
    d_Fy = cuda.to_device(Fy)
    d_den = cuda.to_device(den)
    d_T = cuda.to_device(T)
    d_fl = cuda.to_device(f_l)
    d_f2l = cuda.to_device(f_2l)
    
    pasos = 2000000  # You can edit this line to change the time steps
    tiempo_cuda_1 = time.time()    
    for ii in xrange(pasos):

        mod.momento[blocks, threads](d_f, d_vel_ux, d_vel_uy, d_Fx, d_Fy, d_den, d_fl)         
        mod.energia[blocks, threads](d_g,d_T,d_fl,d_vel_ux,d_vel_uy,d_f2l)                         

        mod.propagacion[blocks, threads](d_f,d_f2)
        mod.propagacion_g[blocks, threads](d_g, d_g2)  

        mod.c_frontera[blocks, threads](d_f)
        mod.c_frontera_g[blocks, threads](d_g)

        mod.cal_den_u_F[blocks, threads](d_f, d_vel_ux, d_vel_uy, d_Fx, d_Fy, d_den, d_fl, d_T)                       
        mod.cal_T_fl_H[blocks, threads](d_g, d_T, d_fl, d_f2l)        

    d_f.to_host()
    d_vel_ux.to_host()
    d_vel_uy.to_host()
    d_g.to_host()  
    d_T.to_host()
    d_den.to_host()
    d_fl.to_host()
    tiempo_cuda_2 = time.time()    

        
#    T = g.sum(axis=2)
    den=f.sum(axis=2)
    t = np.array([tiempo_cuda_2 - tiempo_cuda_1])
    Perfo=[t, lattice_x*lattice_y*pasos/t/1e6]
    
    np.savetxt('den.txt', den,fmt='%.13f')
    np.savetxt('vel_ux.txt', vel_ux,fmt='%.14f')
    np.savetxt('vel_uy.txt', vel_uy,fmt='%.14f')
    np.savetxt('T.txt', T,fmt='%.5f')
    np.savetxt('time.txt',t ,fmt='%.4f') # save arrays in txt
    np.savetxt('Perfo_256X256_300mil.txt',Perfo) # Save performance    
    
#    print "f:", f 
#    print "g:", g
    print "T:", T 
#    print "den_1:", den
#    print "den_2:", f.sum(axis=2) 
#    print "vel_ux", vel_ux
#    print "vel_uy", vel_uy  
    print "T(g.sum):", g.sum(axis=2)    
    print "f_l",f_l
    print"\n MLUPS/2, time =", lattice_x*lattice_y*pasos/t/1e6, "\t Time:", t