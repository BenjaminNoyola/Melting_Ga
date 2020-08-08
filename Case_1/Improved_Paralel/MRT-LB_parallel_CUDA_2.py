# Authors: Benjamín Salomón Noyola García, Suemi Rodriguez Romo

# Numerical Solution 1D performed by MRT-LB, Lattice of 256*8 (difined
# in Parameters.json), the phase change of a bar is modeled in 1D section
# 4.1 of paper: Simulations of Ga melting based on multiple-relaxation time
# lattice Boltzmann method performed with CUDA in Python, Suemi Rodriguez Romo,
# Benjamin Noyola, 2020

# Performed using Python-Numba-CUDA

import numpy as np
from numba import jit, cuda
import math, logging, time, sys
import MRT_LB_shared as d2q5
import get_parser_parameters as gp

logger = logging.getLogger('LOG')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('LOGS_SolNumCUDA.log')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)
"Edit parameters in 'Parameter.json' file"
pr = gp.get_parameters()  # pr is a dictionary of parameters,

def init_sol(g, w_s, T):
    for i in range(pr["lattice_x"]):
        for j in range(pr["lattice_y"]):
            for k in range(5):
                g[i][j][k] = w_s[k]*T[i][j]

if __name__ == '__main__':

    ##################################
    ##### Parameters #####       
    ##################################
    logger.debug("Calculating initial parameters")
    alpha_l = pr["Delta_alfa"] * pr["alpha_s"]  # thermal diffusivity of liquid
    k_s = pr["alpha_s"] * pr["rho"] * pr["Cps"]  # solid thermal conductivity
    k_l = alpha_l * pr["rho"] * pr["Cpl"]  # liquid thermal conductivity
    La = pr["Cpl"] * (pr["T_b"] - pr["T_m"]) / pr["St"]  # latent heat
    H_l = pr["Cpl"] * pr["T_l"] + 1.0 * La  # Enthalpy of liquid  0.04/2 = 0.02,  T_m=0 -> Tl=0.02; fl=1
    H_s = pr["Cps"] * pr["T_s"] + 0.0 * La  # Enthalpy of solid  Ts=-0.02, fl=0
    t = (pr["F_0"] * pr["H"] ** 2) / pr["alpha_s"]  # time step
    c = pr["delta_x"] / pr["delta_t"]  # velocidad de lattice
    c_st = np.sqrt(1.0 / 5.0)  # sound speed of the D2Q9 model
    
    ##################################
    ############# Arrays #############
    ##################################
    logger.debug("Calculating initial arrays")
    T = np.ones([pr["lattice_x"], pr["lattice_y"]])            # temperature is saved in a matrix
    g = np.zeros([pr["lattice_x"], pr["lattice_y"], 5])        # distribution function is saved in a tensor order 3
    h_copia_g = np.zeros([pr["lattice_x"],pr["lattice_y"], 5])  # a copy of g
    f_l = np.zeros([pr["lattice_x"], pr["lattice_y"]])	       # liquid fraction is saved in a matrix
    w_s = np.zeros([5]) 		                 # weight coefficients 
    w_s[0] = (1.0-pr["w_test"])/5.0
    for i in range(1, 5, 1):
        w_s[i] = (4.0+pr["w_test"])/20.
 
    ##################################
    ######## CUDA parameters #########
    ##################################
    logger.debug("Calculating cuda arrays")
    nx, ny, ns = pr["lattice_x"], pr["lattice_y"], 5
    threads = 256, 4 # You can change CUDA parameter
    blocks = (math.ceil(nx/threads[0])+(0!=nx%threads[0]),
              math.ceil(ny/threads[1])+(0!=ny%threads[1]))

    ##################################
    ####### solution begins ##########
    ##################################

    #### initial conditions  ####
    logger.debug("Calculating initial temperature and liquid fraction")
    T = pr["T_i"]*T
    T[0,:], T[pr["lattice_x"]-1,:] = pr["T_b"], pr["T_i"]     # initial temperature
    f_l[0,:], f_l[pr["lattice_x"]-1,:] = 1.0, 0.0 # initial liquid fraction
    init_sol(g, w_s, T)                     # initial solucion (g)

    ### Main CUDA loop  ###
    logger.debug("Send arrays from host to device")
    D_al = np.array([float(pr["Delta_alfa"])])
    d_g = cuda.to_device(g)  # send g from host to device
    copia_g = cuda.to_device(h_copia_g) # send a copy of g from host to device
    d_ws = cuda.to_device(w_s)	# send w_s from host to device
    D_al = cuda.to_device(D_al)	# send thermal diffusivity ratio from host to device
    d_fl = cuda.to_device(f_l) 	# send liquid fraction from host to device

    tiempo_cuda = time.time()   # Measure initial time, before the main CUDA loop begins
    try:
        logger.debug("Running simulation in device")
        print("Running simulation in device")
        for ii in range(pr["steps"]):
            d2q5.collision_local[blocks, threads](d_g, d_fl, D_al) # colision step in local memory
            d2q5.propagacion[blocks, threads](d_g, copia_g)        # streaming step in shared memory
            d2q5.condiciones_frontera[blocks, threads](d_g, d_ws)  # comentar esta línea y descomentar 113-115
    except Exception as ex:
        logger.critical("It was not possible to connect with device")
        print("Error: ", ex)
        sys.exit()

    ###############################################
    ##### Send arrays from device to host
    ###############################################
    logger.debug("Send arrays from device to host")
    d_g.to_host()                   # copy g to host
    d_fl.to_host()                  # copy liquid fraction (f_l) to host

    ###############################################
    #### Calculation of simulation results
    ###############################################
    T = g.sum(axis=2)               # calculate temperature from the distribution function
    t = time.time() - tiempo_cuda   # Measure final time, after the main CUDA loop ends
    print("\n time", t, " [s]")             # print time of simulaton
    print("\n MLUPS/2 = ", nx*ny*pr["steps"]/t/1e6) # print million of lattice update per second (MLUPS)
    Perfo = [t, nx*ny*pr["steps"]/t/1e6]    # simulation time, MLUPS
    T = np.transpose(T)                     # Transpose of temperature array
    f_l = np.transpose(f_l)                 # Transpose of liquid fraction array
    print("\n Temperature: \n", T)          # Print temperature as a matrix, every cell corresponds to a lattice
    print("\n Liquid fraction \n", f_l)     # Print Liquid fraction as a matrix, every cell corresponds to a lattice

    ################################################
    #### Save results of simulation in txt format
    ################################################
    np.savetxt('Perfo_.txt', Perfo)         # Save in txt the performance
    np.savetxt('T_.txt', T, fmt='%.6f')     # Save in txt the temperature
    np.savetxt('f_l_.txt', f_l, fmt='%.3f') # Save in txt liquid fraction
