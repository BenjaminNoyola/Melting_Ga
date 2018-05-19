#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Authors: Benjamín S. Noyola García, Suemi Rodriguez Romo

"""Numerical solution of Solid-liquid phase change simulation of a material (Ga) 
immersed in a porous media, performed by Numba-CUDA. This simulation uses MRT-LBM, 
with a D2Q5 stencil, for the heat transfer in Ga and a D2Q9 stencil for the momentum 
transfers in the liquid Ga. The last one takes into account  natural convection.
This software is initially inspired in the article: Qing, Liu, Ya-Ling He*. 
Double multiple-relaxation-time lattice Boltzmann Model for solid-liquid phase 
change with natural convection in porous media, Physica A. 438(2015) 94-106.
This program performs the simulation from an porosity image""" 

import numpy as np
import scipy.misc, sys, random
from numpy import linalg as LA
from numba import jit,cuda
import numba, time
import matplotlib.pyplot as plt
import modulos as mod   
         
if __name__ == '__main__':
#########################################################################################
############################### parameters ############################################
#########################################################################################
    lattice_x , lattice_y = 256, 256 # Lattices 
    Ra = 8.409e5 	# Rayleight number
    J = 1.0         # viscosity ratio
    Pr = 0.0208 	# Prandtl number
    dm = 25.211188820977277 # particle diameter
    Lambda = 0.2719	# termal efective difusivity ratio respect to the liquid difusivity
    T_h = 45.0		# hot temperature
    T_c = 20.0		# cold temperature
    Cp=Cpl=Cps = 1.0	# specific heat
    T_m = 29.78		# melting temperature
    T_i = T_o = 20.0# initial temperature
    H = float(lattice_y)	# characteristic lenght
    Fo = 1.829       # Fourier number
    St = 0.1241	    # Stephan number
    Ma = 0.1 		# Mach number
    sigma_l = 0.8604 # termal capacity ratio in liquid
    sigma_s = 0.8352 # termal capacity ratio in solid
    La = Cpl*(T_h-T_c)/St # latent heat
    rho_o = 1.0 	 # initial density
    w_test = -2.0	
    delta = delta_x = delta_y = delta_t=1.0 # space and time step
    c = delta_x/delta_t
    c_st = np.sqrt(1.0/5.0)
    c_s = c/np.sqrt(3.0)
    H_l = Cpl*(T_m+0.5) + 1.0*La # liquid enthalpy   
    H_s = Cps*(T_m-0.5) + 0.0*La # Solid enthalpy
    t=int((Fo*H)/0.08731)
    tau_v = 0.5


#################################################################################
############Calcultation of porosity and permeability from images ################
################################################################################# 
    medio = scipy.misc.imread('m_poro_.png',flatten=True) 
    #~ scipy.misc.imsave('medio.png', medio) 
    np.savetxt('medio_no-binario.dat', medio,fmt='%.2f') # Save binary image 1=pore; 0=solid
    
######################################################################################
##################### statical informatio of the image ###############################
######################################################################################
#    print "\nImage size: ", medio.size
#    print "Image shape: ", medio.shape
    nx,ny = medio.shape    
    size_im = nx*ny        
#    print " Maximum value in image: ", medio.max()
#    print " Minimum value in image: ", medio.min()
#    print " Mean value in image: ", medio.mean()
    
    #######################################################################################
    ################################## Porosity calculation ###############################
    #######################################################################################
    umbral = 90.0      # before 90 in gray scale is a pore.
    for i in range(nx):
        for j in range(ny):
            if medio[i,j] <= umbral: # 1 = pore
                medio[i,j]= 1.0
            else:
                medio[i,j]=0.0       # 0 = solid
    
    porosidad = np.sum(medio)/size_im  # porosity = (sum of pores)/total elements
#    print "\n global porosity: ", porosidad
    
    ##########################################################################################
    ######################## porous media image, uncoment #################################
    ##########################################################################################
#    import matplotlib.pyplot as plt  # 
#    plt.gray()                       #
#    plt.imshow(medio)
#    plt.show()
    
    ##########################################################################################
    ############### calculate porosity matrix #################
    ##########################################################################################
    part_x = 4
    part_y = 4  
    div_x = nx//part_x  
    div_y = ny//part_y 
    
    if nx % part_x != 0 or ny % part_y != 0:    
        print "\ncorrija (part_x, part_y) para qu sea divisible entre ",nx, ny
        sys.exit(1) 
         
    porosidades=np.zeros([part_x, part_y]) # se creo una matriz vacía para llenar la porosidad
    
    m,init_1 = 0, div_x  
    for i in range (0,nx, div_x):
        n, init_2 = 0, div_y
        for j in range (0,ny, div_y):
            porosidades[m,n] = (medio[i:init_1, j:init_2].sum()) / float((nx*ny)/(part_x*part_y)) # se encuentra la porosidad de la celda
            if porosidades[m,n]==1.0:
                porosidades[m,n]=random.uniform(0.9, 0.99)
            init_2 = init_2+div_y
            n = n+1
        init_1 = init_1+div_x
        m = m+1
#    print "\n Matriz de porosidades: \n",porosidades 
#    print "\n promedio de las porosidades (matriz): ",porosidades.mean() # el promedio de la matriz de porosidades recupera la porosidad global.
    np.savetxt('porosidades.dat', porosidades,fmt='%.5f')
    
    ##########################################################################################
    ##################### Mapout the porosity matrix to the domain size. #####################
    ##########################################################################################
    
    poros = np.zeros([lattice_x, lattice_y])
    Ks = np.zeros([lattice_x, lattice_y])
    
    steep_x=lattice_x/part_x
    steep_y=lattice_y/part_y
    
    k2=0                                    # complete porosity matrix
    for i in range(0,lattice_y,steep_y):
        k1=0
        for j in range(0,lattice_x,steep_x):   
            for k in range(steep_y):
                for l in range(steep_x):
                     poros[j+l, i+k] = porosidades[k1,k2]               
            k1 = k1+1
        k2 = k2+1
    
    for i in range(lattice_y):              # complete permeability matriz
        for j in range(lattice_x):
            Ks[i,j]=(poros[i,j]**3 * dm**2)/(175*(1.0-poros[i,j])**2)  #Ks, matriz de permeabilidades, completa
    
    np.savetxt('poros.dat', poros,fmt='%.5f')
    np.savetxt('K.dat', Ks,fmt='%.5f')
#    param = array([T_h, T_c, Ra, Da, poros, J, Pr, Lambda, T_m, T_i, H, St, Ma, sigma_l, sigma_s, La, rho_o, w_test, delta, c, c_st, c_s, K, H_l, H_s])
#                  [0,   1,   2,  3,   4,    5, 6,  7,       8,   9,  10,11, 12, 13,      14,      15, 16,     17,     18,   19, 20,  21,  22, 23, 24]

#########################################################################################
############ 2.- ........temperature field array:...........................############
#########################################################################################
    T = np.ones([lattice_x, lattice_y]) 
    g = np.zeros([lattice_x, lattice_y, 5]) 
    g_eq = np.zeros([lattice_x, lattice_y, 5]) 
    H_k  = np.zeros([lattice_x,lattice_y])			
    f_l  = np.zeros([lattice_x,lattice_y])			
    w_s = np.zeros([5]) 		#pesos_s 
    
#########################################################################################
############ 3.- ........velocity field array:...........................############
#########################################################################################
    den = np.ones([lattice_x,lattice_y]) 
    f = np.zeros([lattice_x,lattice_y,9]) 
    f_eq = np.zeros([lattice_x,lattice_y,9]) 
    w = np.zeros([9]) 		
    s = np.array([1.0, 1.1, 1.1, 1.0, 1.2, 1.0, 1.2, 1.0/tau_v, 1.0/tau_v]) #construye la matriz de relajación
    S = np.zeros([9])
    relax_f = np.zeros([9,9])   
    np.fill_diagonal(relax_f,s) 
    vel_vx = np.zeros([lattice_x,lattice_y]) 
    vel_vy = np.zeros([lattice_x,lattice_y]) 
    vel_ux = np.zeros([lattice_x,lattice_y]) 
    vel_uy = np.zeros([lattice_x,lattice_y]) 
    Fx = np.zeros([lattice_x,lattice_y]) 
    Fy = np.zeros([lattice_x,lattice_y]) 
    
    # weights (temperature field):
    for k in range(5):
        if k == 0:
            w_s[k] = (1.0-w_test)/5.0
        else:
            w_s[k] = (4.0+w_test)/20.0
    
    # weights (velocity field):
    for k in range(9):
        if k == 0:
            w[k] = 4./9.
        elif (k >= 1 and k <= 4):
            w[k] = 1./9.
        else:
            w[k] = 1./36.
    
# 3.- ...............Initial conditions.....................................
    # Temperature
    T=T_i*T
    for j in range (lattice_y):
        T[0,j]=T_h
        T[lattice_x-1,j] = T_c
        f_l[0,j]=1.0                # liquid fraction
        f_l[lattice_x-1,j] =0.0
    f_2l = np.copy(f_l)
    
    # enthalpy calculation 
    for i in xrange(lattice_x):
        for j in xrange(lattice_y):
            H_k[i,j] = Cps*T[i,j] + f_l[i,j]*La
    
#  4.- ............Initial distribution function....................
    # equilibriun distribution function
    for i in range(lattice_x):
        for j in range(lattice_y):
            for k in range(5):
                g_eq[i,j,k] = w_s[k]*T[i,j]
    g=np.copy(g_eq)
    g2=np.copy(g_eq)    
    # equilibrium distribution function (velocity field)
    for i in range(lattice_x):
        for j in range(lattice_y):
            for k in xrange(9):
                f_eq[i,j,k] = w[k]*den[i,j]         
    f=np.copy(f_eq)
    f2=np.copy(f_eq)
    ##################################
    ######## CUDA parameters #########       
    ##################################
    threads = 256,1 #1024#512
    blocks = (lattice_x/threads[0]+(0!=lattice_x%threads[0]),
              lattice_y/threads[1]+(0!=lattice_y%threads[1]) )
#    print threads
#5.-  Send arrays to device ###
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
    d_poros = cuda.to_device(poros)
    d_Ks = cuda.to_device(Ks)
    
    pasos = 2000000    # you could edit this line to change the time steps
    tiempo_cuda_1 = time.time()    
    for ii in xrange(pasos):

        mod.momento[blocks, threads](d_f, d_vel_ux, d_vel_uy, d_Fx, d_Fy, d_den, d_fl, d_poros)         
        mod.energia[blocks, threads](d_g,d_T,d_fl,d_vel_ux,d_vel_uy,d_f2l, d_poros)                         

        mod.propagacion[blocks, threads](d_f,d_f2)
        mod.propagacion_g[blocks, threads](d_g, d_g2)  

        mod.c_frontera[blocks, threads](d_f)
        mod.c_frontera_g[blocks, threads](d_g)

        mod.cal_den_u_F[blocks, threads](d_f, d_vel_ux, d_vel_uy, d_Fx, d_Fy, d_den, d_fl, d_T, d_poros, d_Ks)                       
        mod.cal_T_fl_H[blocks, threads](d_g, d_T, d_fl, d_f2l)        

    d_f.to_host()
    d_vel_ux.to_host()
    d_vel_uy.to_host()
    d_g.to_host()  
    d_T.to_host()
    d_den.to_host()
    d_fl.to_host()
    tiempo_cuda_2 = time.time()    

    strf=np.zeros([lattice_x,lattice_y])
    strf[0,0]=0.0
    for j in range(lattice_y):
        rhoav=0.5*(den[0,j-1]+den[0,j])
        if j != 0.0: strf[0,j] = strf[0,j-1]-rhoav*0.5*(vel_uy[0,j-1]+vel_uy[0,j])
        for i in range(1,lattice_x):
            rhom=0.5*(den[i,j]+den[i-1,j])
            strf[i,j]=strf[i-1,j]+rhom*0.5*(vel_ux[i-1,j]+vel_ux[i,j])
    
    strf2=np.zeros([lattice_x,lattice_y])
    strf2[0,0]=0.0
    for j in range(lattice_x):
        rhoav2=0.5*(den[j-1,0]+den[j,0])
        if j != 0.0: strf2[j,0] = strf2[j-1,0]-rhoav2*0.5*(vel_uy[j-1,0]+vel_uy[j,0])
        for i in range(1,lattice_y):
            rhom=0.5*(den[j,i]+den[j,i-1])
            strf2[j,i]=strf2[j,i-1]+rhom*0.5*(vel_ux[j,i-1]+vel_ux[j,i])
        
#    T = g.sum(axis=2)
    den=f.sum(axis=2)
    t = np.array([tiempo_cuda_2 - tiempo_cuda_1])
    mlups = np.array([lattice_x*lattice_y*pasos/t/1e6])
    Perfo=[t, lattice_x*lattice_y*pasos/t/1e6]  # simulation time, MLUPS
   
    np.savetxt('den.txt', den,fmt='%.13f')       # save the density in a .txt file
    np.savetxt('vel_ux.txt', vel_ux,fmt='%.14f') # save the velocity x
    np.savetxt('vel_uy.txt', vel_uy,fmt='%.14f') # save the velocity y
#    np.savetxt('strf.txt', strf,fmt='%.14f')
#    np.savetxt('strf2.txt', strf,fmt='%.14f')
    np.savetxt('T.txt', T,fmt='%.5f')      # save temperature in a .txt file
    np.savetxt('Performance_256X8_2M.txt',Perfo) # Save performance
#    print "f:", f 
#    print "g:", g
    print "T:", T 
    print "den_1:", den
#    print "den_2:", f.sum(axis=2) 
#    print "vel_ux", vel_ux
#    print "vel_uy", vel_uy  
#    print "T(g.sum):", g.sum(axis=2)    
    print "f_l",f_l
    print"\n mlups/2:", lattice_x*lattice_y*pasos/t/1e6, "\t Time:", t