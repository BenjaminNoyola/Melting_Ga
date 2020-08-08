#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Authors: Benjamín S. Noyola García, Suemi Rodriguez Romo

"""Numerical solution of Solid-liquid phase change simulation of a
material (Ga) immersed in a porous media, performed by Numba-CUDA.
This simulation uses MRLBM, with a D2Q5 stencil, for the heat transfer
in Ga and a D2Q9 stencil for the momentum transfers in the liquid Ga.
The last one takes into account  natural convection.
This software is initially inspired in the paper: Simulations of Ga
melting based on multiple-relaxation time lattice Boltzmann method
performed with CUDA in Python, Suemi Rodriguez Romo, Benjamin Noyola,
2020. This program solves the simulation with homogeneous porosity"""

import logging
import numpy as np
from numba import jit,cuda
import numba, time, math
import modulos as mod
import get_parser_parameters as gp

logger = logging.getLogger('LOG')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('LOGS_SolNumCUDA_2.log')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

pr = gp.get_parameters()  # pr is a dictionary of parameters,
if __name__ == '__main__':

    #######################################################
    # 1.- Parameters
    #######################################################
    logger.debug("Calculate parameters")
    La = pr["Cpl"]*(pr["T_h"]-pr["T_c"])/pr["St"]
    delta_x = delta_y = delta_t=1.0 # space and time step
    c = delta_x/delta_t
    c_st = np.sqrt(1.0/5.0)
    c_s = c/np.sqrt(3.0)
    tau_v = 0.5 + (0.1*pr["H"]*np.sqrt(3.0*pr["Pr"]))/(np.sqrt(pr["Ra"]))
    K = pr["Da"]*pr["H"]**2         # Permeability
    H_l = pr["Cpl"]*(pr["T_m"]+0.05) + 1.0*La # liquid enthalpy  0.04/2 = 0.02,  T_m=0 -> Tl=0.02
    H_s = pr["Cps"]*(pr["T_m"]-0.05) + 0.0*La # solid enthalpy  Ts=-0.02
    # t = int((pr["Fo"]*pr["H"]**2)/0.1117571953522575)  #/0.08731)

    ###################################################
    # 2.- Initial arrays
    ###################################################
    # Temperature field arrays:
    logger.debug("Initial arrays of temperature field")
    T = np.ones([pr["lattice_x"], pr["lattice_y"]])     # temperature
    g = np.zeros([pr["lattice_x"], pr["lattice_y"], 5]) # distribution function
    g_eq = np.zeros([pr["lattice_x"], pr["lattice_y"], 5]) # equilibrium distribution function
    H_k  = np.zeros([pr["lattice_x"],pr["lattice_y"]])	# enthalpy
    f_l  = np.zeros([pr["lattice_x"],pr["lattice_y"]])	# liquid fraction
    w_s = np.zeros([5]) 		                        # weights corresponding to d2q5
    
    # velocity field arrays:
    logger.debug("Initial arrays of velocity field")
    den = np.ones([pr["lattice_x"],pr["lattice_y"]])    # density
    f = np.zeros([pr["lattice_x"],pr["lattice_y"],9])   # distribution function
    f_eq = np.zeros([pr["lattice_x"],pr["lattice_y"],9]) # equilibrium distribution function
    w = np.zeros([9]) 		                            # weights corresponding to d2q9
    s = np.array([1.0, 1.1, 1.1, 1.0, 1.2, 1.0, 1.2, 1.0/tau_v, 1.0/tau_v]) 
    S = np.zeros([9])
    relax_f = np.zeros([9,9])
    np.fill_diagonal(relax_f,s) 
    vel_vx = np.zeros([pr["lattice_x"],pr["lattice_y"]])
    vel_vy = np.zeros([pr["lattice_x"],pr["lattice_y"]])
    vel_ux = np.zeros([pr["lattice_x"],pr["lattice_y"]])
    vel_uy = np.zeros([pr["lattice_x"],pr["lattice_y"]])
    Fx = np.zeros([pr["lattice_x"],pr["lattice_y"]])
    Fy = np.zeros([pr["lattice_x"],pr["lattice_y"]])
    
    # weights(temperature field):
    logger.debug("Calculate weights of temperature field")
    for k in range(5):
        if k == 0:
            w_s[k] = (1.0-pr["w_test"])/5.0
        else:
            w_s[k] = (4.0+pr["w_test"])/20.0
    
    # weight(velocity field):
    logger.debug("Calculate weights of velocity field")
    for k in range(9):
        if k == 0:
            w[k] = 4./9.
        elif (k >= 1 and k <= 4):
            w[k] = 1./9.
        else:
            w[k] = 1./36.

    ########################################################
    # 3.- Initial conditions
    ########################################################
    # temperature field:
    logger.debug("Initial temperature and liquid fraction arrays")
    T = pr["T_o"]*T
    for j in range (pr["lattice_y"]):
        T[0,j]=pr["T_h"]
        T[pr["lattice_x"]-1,j] = pr["T_c"]
        f_l[0,j]=1.0
        f_l[pr["lattice_x"]-1,j] =0.0
    f_2l = np.copy(f_l)
    
    # Enthalpy calculation (temperature field)
    logger.debug("Calculate enthalpy array")
    for i in range(pr["lattice_x"]):
        for j in range(pr["lattice_y"]):
            H_k[i, j] = pr["Cps"]*T[i, j] + f_l[i, j] * La

    ##################################################
    #  4.- initial distribution function
    ##################################################
    logger.debug("Initial distribution function")
    for i in range(pr["lattice_x"]):
        for j in range(pr["lattice_y"]):
            for k in range(5):
                g_eq[i, j, k] = w_s[k] * T[i, j]  # temperature field
    g = np.copy(g_eq)
    g2 = np.copy(g_eq)

    for i in range(pr["lattice_x"]):
        for j in range(pr["lattice_y"]):
            for k in range(9):
                f_eq[i,j,k] = w[k]*den[i,j] # velocity field
    f = np.copy(f_eq)
    f2 = np.copy(f_eq)

    ###########################################
    ######## parámetros de CUDA
    ###########################################
    logger.debug("parameter of CUDA")
    threads = 256,2 #1024#512
    blocks = (math.ceil(pr["lattice_x"]/threads[0]) + (0 != pr["lattice_x"] % threads[0]),
              math.ceil(pr["lattice_y"]/threads[1]) + (0 != pr["lattice_y"] % threads[1]))

    #############################################
    #5.-  comienza ciclo, envia calculo a CUDA
    #############################################
    logger.debug("Send arrays from host to device")
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

    time_cuda_1 = time.time()
    print("Running Simulation")
    logger.debug("Start Simulation")

    for ii in range(pr["steps"]):
        mod.momento[blocks, threads](d_f, d_vel_ux, d_vel_uy, d_Fx, d_Fy, d_den, d_fl)         
        mod.energia[blocks, threads](d_g,d_T,d_fl,d_vel_ux,d_vel_uy,d_f2l)                         

        mod.propagacion[blocks, threads](d_f,d_f2)
        mod.propagacion_g[blocks, threads](d_g, d_g2)  

        mod.c_frontera[blocks, threads](d_f)
        mod.c_frontera_g[blocks, threads](d_g)

        mod.cal_den_u_F[blocks, threads](d_f, d_vel_ux, d_vel_uy, d_Fx, d_Fy, d_den, d_fl, d_T)                       
        mod.cal_T_fl_H[blocks, threads](d_g, d_T, d_fl, d_f2l)

    logger.debug("Send results from device to host")
    d_f.to_host()
    d_vel_ux.to_host()
    d_vel_uy.to_host()
    d_g.to_host()
    d_T.to_host()
    d_den.to_host()
    d_fl.to_host()
    time_cuda_2 = time.time()

    den = f.sum(axis=2)
    t = np.array([time_cuda_2 - time_cuda_1])

    ###################################################
    # Save results
    ###################################################

    Perfo = [t, pr["lattice_x"]*pr["lattice_y"]*pr["steps"]/t/1e6]
    logger.debug("Save arrays of solution in txt format")
    np.savetxt('Results/den.txt', den,fmt='%.13f')
    np.savetxt('Results/vel_ux.txt', vel_ux,fmt='%.14f')
    np.savetxt('Results/vel_uy.txt', vel_uy,fmt='%.14f')
    np.savetxt('Results/T_.txt', T,fmt='%.5f')
    np.savetxt('Results/time_.txt', t, fmt='%.4f') # save arrays in txt
    np.savetxt('Results/f_l_.txt', f_l, fmt='%.4f') # save arrays in txt
    np.savetxt('Results/Performance_.txt', Perfo) # Save performance

#    print "f:", f 
#    print "g:", g
#    print "den_1:", den
#    print "den_2:", f.sum(axis=2) 
#    print "vel_ux", vel_ux
#    print "vel_uy", vel_uy  
    print("T:", T)
    print("T(g.sum):", g.sum(axis=2))
    print("f_l", f_l)
    print("\n MLUPS/2, time =", pr["lattice_x"]*pr["lattice_y"]*pr["steps"]/t/1e6, "\t Time:", t)
