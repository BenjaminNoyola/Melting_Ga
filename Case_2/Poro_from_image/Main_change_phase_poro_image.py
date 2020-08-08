
# Authors: Benjamín S. Noyola García, Suemi Rodriguez Romo

"""Numerical solution of Solid-liquid phase change simulation of a
material (Ga) immersed in a porous media, performed by Numba-CUDA.
This simulation uses MRT-LBM, with a D2Q5 stencil, for the heat
transfer in Ga and a D2Q9 stencil for the momentum transfers in
the liquid Ga. The last one takes into account  natural convection.
This software is inspired in the article: Simulations of Ga
melting based on multiple-relaxation time lattice Boltzmann method
performed with CUDA in Python, Suemi Rodriguez Romo, Benjamin Noyola,
2020. This program performs the simulation from an porosity image"""

import numpy as np
import scipy.misc, sys, random
from numpy import linalg as LA
from numba import jit,cuda
import numba, time, math, cv2
import matplotlib.pyplot as plt
import modulos as mod   
import logging
import imageio
import get_parser_parameters as gp
from PIL import Image

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
#########################################################################################
############################### parameters ############################################
#########################################################################################
    La = pr["Cpl"]*(pr["T_h"] - pr["T_c"])/pr["St"] # latent heat
    c = pr["delta_x"]/pr["delta_t"]
    c_st = np.sqrt(1.0/5.0)
    c_s = c/np.sqrt(3.0)
    H_l = pr["Cpl"]*(pr["T_m"]+0.5) + 1.0 * La # liquid enthalpy
    H_s = pr["Cps"]*(pr["T_m"]-0.5) + 0.0 * La # Solid enthalpy
    t = int((pr["Fo"]*pr["H"])/0.08731)
    tau_v = 0.5 + (0.1*pr["H"]*np.sqrt(3.0*pr["Pr"]))/(np.sqrt(pr["Ra"]))

#################################################################################
############Calcultation of porosity and permeability from images ################
#################################################################################
    medio = cv2.imread("m_poro_.png", cv2.IMREAD_GRAYSCALE)
    print(np.shape(medio))
    #~ scipy.misc.imsave('medio.png', medio)
    np.savetxt('Results/medio_no-binario.dat', medio, fmt='%.2f') # Save binary image 1=pore; 0=solid
    
######################################################################################
##################### statical informatio of the image ###############################
######################################################################################
#    print "\nImage size: ", medio.size
#    print "Image shape: ", medio.shape
    nx, ny = medio.shape
    size_im = nx*ny        
#    print " Maximum value in image: ", medio.max()
#    print " Minimum value in image: ", medio.min()
#    print " Mean value in image: ", medio.mean()
    
    #######################################################################################
    ################################## Porosity calculation ###############################
    #######################################################################################

    for i in range(nx):
        for j in range(ny):
            if medio[i, j] <= pr["threshold"]: # 1 = pore
                medio[i, j] = 1.0
            else:
                medio[i, j] = 0.0       # 0 = solid
    
    # porosidad = np.sum(medio)/size_im  # porosity = (sum of pores)/total elements
    # print "\n global porosity: ", porosidad
    #

    ##########################################################################################
    ############### calculate porosity matrix #################
    ##########################################################################################

    div_x = nx//pr["part_x"]
    div_y = ny//pr["part_y"]
    
    if nx % pr["part_x"] != 0 or ny % pr["part_y"] != 0:
        print("""\nChoose another number of partitions in x and/or y axis 
        in the Parameters of configuration, The dimensions of the domain 
        must be divisible with the number of partitions""", " Shape of domain:",nx, " X ", ny)
        sys.exit(1)
         
    porosidades = np.zeros([pr["part_x"], pr["part_y"]]) # se creo una matriz vacía para llenar la porosidad
    
    m,init_1 = 0, div_x  
    for i in range (0,nx, div_x):
        n, init_2 = 0, div_y
        for j in range (0,ny, div_y):
            porosidades[m,n] = (medio[i:init_1, j:init_2].sum()) / float((nx*ny)/(pr["part_x"] * pr["part_y"])) # se encuentra la porosidad de la celda
            if porosidades[m,n]==1.0:
                porosidades[m,n]=random.uniform(0.9, 0.99)
            init_2 = init_2+div_y
            n = n+1
        init_1 = init_1+div_x
        m = m+1
    np.savetxt('Results/porosidades.dat', porosidades,fmt='%.5f')
    
    ##########################################################################################
    ##################### Mapout the porosity matrix to the domain size. #####################
    ##########################################################################################
    
    poros = np.zeros([pr["lattice_x"], pr["lattice_y"]])
    Ks = np.zeros([pr["lattice_x"], pr["lattice_y"]])
    
    steep_x = int(pr["lattice_x"]/pr["part_x"])
    steep_y = int(pr["lattice_y"]/pr["part_y"])
    
    k2=0                                    # complete porosity matrix
    for i in range(0, pr["lattice_y"], steep_y):
        k1=0
        for j in range(0, pr["lattice_x"], steep_x):
            for k in range(steep_y):
                for l in range(steep_x):
                     poros[j+l, i+k] = porosidades[k1,k2]               
            k1 = k1+1
        k2 = k2+1
    
    for i in range(pr["lattice_y"]):              # complete permeability matriz
        for j in range(pr["lattice_x"]):
            Ks[i,j]=(poros[i,j]**3 * pr["dm"]**2)/(175*(1.0-poros[i,j])**2)  #Ks, matriz de permeabilidades, completa
    
    np.savetxt('Results/poros.dat', poros,fmt='%.5f')
    np.savetxt('Results/K.dat', Ks,fmt='%.5f')

#########################################################################################
############ 2.- ........temperature field array:...........................############
#########################################################################################
    T = np.ones([pr["lattice_x"], pr["lattice_y"]])
    g = np.zeros([pr["lattice_x"], pr["lattice_y"], 5])
    g_eq = np.zeros([pr["lattice_x"], pr["lattice_y"], 5])
    H_k  = np.zeros([pr["lattice_x"], pr["lattice_y"]])
    f_l  = np.zeros([pr["lattice_x"], pr["lattice_y"]])
    w_s = np.zeros([5]) 		#pesos_s 
    
#########################################################################################
############ 3.- ........velocity field array:...........................############
#########################################################################################
    den = np.ones([pr["lattice_x"], pr["lattice_y"]])
    f = np.zeros([pr["lattice_x"], pr["lattice_y"], 9])
    f_eq = np.zeros([pr["lattice_x"], pr["lattice_y"], 9])
    w = np.zeros([9]) 		
    s = np.array([1.0, 1.1, 1.1, 1.0, 1.2, 1.0, 1.2, 1.0/tau_v, 1.0/tau_v]) #construye la matriz de relajación
    S = np.zeros([9])
    relax_f = np.zeros([9,9])   
    np.fill_diagonal(relax_f,s) 
    vel_vx = np.zeros([pr["lattice_x"], pr["lattice_y"]])
    vel_vy = np.zeros([pr["lattice_x"], pr["lattice_y"]])
    vel_ux = np.zeros([pr["lattice_x"], pr["lattice_y"]])
    vel_uy = np.zeros([pr["lattice_x"], pr["lattice_y"]])
    Fx = np.zeros([pr["lattice_x"], pr["lattice_y"]])
    Fy = np.zeros([pr["lattice_x"], pr["lattice_y"]])
    
    # weights (temperature field):
    for k in range(5):
        if k == 0:
            w_s[k] = (1.0 - pr["w_test"])/5.0
        else:
            w_s[k] = (4.0 + pr["w_test"])/20.0
    
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
    T = pr["T_o"] * T
    for j in range (pr["lattice_y"]):
        T[0,j] = pr["T_h"]
        T[pr["lattice_x"]-1,j] = pr["T_c"]
        f_l[0,j] = 1.0                # liquid fraction
        f_l[pr["lattice_x"] - 1, j] = 0.0
    f_2l = np.copy(f_l)
    
    # enthalpy calculation 
    for i in range(pr["lattice_x"]):
        for j in range(pr["lattice_y"]):
            H_k[i,j] = pr["Cps"] * T[i,j] + f_l[i,j]*La
    
#  4.- ............Initial distribution function....................
    # equilibriun distribution function
    for i in range(pr["lattice_x"]):
        for j in range(pr["lattice_y"]):
            for k in range(5):
                g_eq[i, j, k] = w_s[k] * T[i, j]
    g = np.copy(g_eq)
    g2 = np.copy(g_eq)
    # equilibrium distribution function (velocity field)
    for i in range(pr["lattice_x"]):
        for j in range(pr["lattice_y"]):
            for k in range(9):
                f_eq[i, j, k] = w[k] * den[i,j]
    f = np.copy(f_eq)
    f2 = np.copy(f_eq)
    # ##################################
    # ######## CUDA parameters #########
    # ##################################
    # threads = 256,1 #1024#512
    # blocks = (lattice_x/threads[0]+(0!=lattice_x%threads[0]),
    #           pr["lattice_y"]/threads[1]+(0!=pr["lattice_y"]%threads[1]) )

    ###########################################
    ######## parámetros de CUDA
    ###########################################
    logger.debug("parameter of CUDA")
    threads = 256, 1  # 1024#512
    blocks = (math.ceil(pr["lattice_x"] / threads[0]) + (0 != pr["lattice_x"] % threads[0]),
              math.ceil(pr["lattice_y"] / threads[1]) + (0 != pr["lattice_y"] % threads[1]))

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
    
    # pr["steps"] = 2000000    # you could edit this line to change the time steps
    tiempo_cuda_1 = time.time()    
    for ii in range(pr["steps"]):

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

    strf=np.zeros([pr["lattice_x"],pr["lattice_y"]])
    strf[0,0]=0.0
    for j in range(pr["lattice_y"]):
        rhoav=0.5*(den[0,j-1]+den[0,j])
        if j != 0.0: strf[0,j] = strf[0,j-1]-rhoav*0.5*(vel_uy[0,j-1]+vel_uy[0,j])
        for i in range(1,pr["lattice_x"]):
            rhom=0.5*(den[i,j]+den[i-1,j])
            strf[i,j]=strf[i-1,j]+rhom*0.5*(vel_ux[i-1,j]+vel_ux[i,j])
    
    strf2=np.zeros([pr["lattice_x"],pr["lattice_y"]])
    strf2[0,0]=0.0
    for j in range(pr["lattice_x"]):
        rhoav2=0.5*(den[j-1,0]+den[j,0])
        if j != 0.0: strf2[j,0] = strf2[j-1,0]-rhoav2*0.5*(vel_uy[j-1,0]+vel_uy[j,0])
        for i in range(1,pr["lattice_y"]):
            rhom=0.5*(den[j,i]+den[j,i-1])
            strf2[j,i]=strf2[j,i-1]+rhom*0.5*(vel_ux[j,i-1]+vel_ux[j,i])
        
#    T = g.sum(axis=2)
    den=f.sum(axis=2)
    t = np.array([tiempo_cuda_2 - tiempo_cuda_1])
    mlups = np.array([pr["lattice_x"]*pr["lattice_y"]*pr["steps"]/t/1e6])
    Perfo=[t, pr["lattice_x"]*pr["lattice_y"]*pr["steps"]/t/1e6]  # simulation time, MLUPS
   
    np.savetxt('Results/den.txt', den,fmt='%.13f')       # save the density in a .txt file
    np.savetxt('Results/vel_ux.txt', vel_ux,fmt='%.14f') # save the velocity x
    np.savetxt('Results/vel_uy.txt', vel_uy,fmt='%.14f') # save the velocity y
#    np.savetxt('Results/strf.txt', strf,fmt='%.14f')
#    np.savetxt('Results/strf2.txt', strf,fmt='%.14f')
    np.savetxt('Results/T_.txt', T,fmt='%.5f')      # save temperature in a .txt file
    np.savetxt('Results/Performance_256X8_2M.txt',Perfo) # Save performance
#    print "f:", f 
#    print "g:", g
    print("T:", T)
    print("den_1:", den)
#    print "den_2:", f.sum(axis=2) 
#    print "vel_ux", vel_ux
#    print "vel_uy", vel_uy  
#    print "T(g.sum):", g.sum(axis=2)    
    print("f_l",f_l)
    print("\n mlups/2:", pr["lattice_x"] * pr["lattice_y"] * pr["steps"]/t/1e6, "\t Time:", t)