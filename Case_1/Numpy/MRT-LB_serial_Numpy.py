# Autors: Benjamín S. Noyola García, Suemi Rogriguez Romo
# 
# Numerical Solution 1D performed by MRT-LB, Lattice of 256*8,
# the phase change of a bar is modeled in 1D section 4.1 of 
# paper: Simulations of Ga melting based on multiple-relaxation time
# lattice Boltzmann method performed with CUDA in Python, Suemi Rodriguez Romo,
# Benjamin Noyola, 2020
#
# Performed with python-numpy
import numpy as np
import Inicializacion as ini
import time, logging
import get_parser_parameters as gp

logger = logging.getLogger('LOG')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('LOGS_SolNum.log')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

if __name__ == '__main__':
	pr = gp.get_parameters()
	logger.debug("Start load and calculation of parameters")
	alpha_l= pr["Delta_alfa"] * pr["alpha_s"]  # thermal diffusivity of liquid
	k_s= pr["alpha_s"] * pr["rho"] * pr["Cps"]  # solid thermal conductivity
	k_l= alpha_l * pr["rho"] * pr["Cpl"]  # liquid thermal conductivity
	La= pr["Cpl"] * (pr["T_b"] - pr["T_m"]) / pr["St"]  # latent heat
	H_l= pr["Cpl"] * pr["T_l"] + 1.0 * La  # Enthalpy of liquid  0.04/2 = 0.02,  T_m=0 -> Tl=0.02; fl=1
	H_s= pr["Cps"] * pr["T_s"] + 0.0 * La  # Enthalpy of solid  Ts=-0.02, fl=0
	t= (pr["F_0"] * pr["H"] ** 2) / pr["alpha_s"]  # time step
	c= pr["delta_x"] / pr["delta_t"]  # velocidad de lattice
	c_st= np.sqrt(1.0 / 5.0)  # sound speed of the D2Q9 model
	g = ini.distribution_func()
	f_l = ini.fl()
	H_k = ini.enthalpy()
	N = ini.N_()
	N_inv = ini.N_inv()
	w_s = ini.weights()
	############################################################
	# Simulation begins
	###########################################################
	tiempo_inicio = time.time()
	logger.debug("Simulation begins")
	n_res1 = np.zeros([5])  # distribution function after collision
	alpha_e = np.zeros([pr["lattice_y"], pr["lattice_x"]])  # thermal diffusivity
	tau_t = np.zeros([pr["lattice_y"], pr["lattice_x"]])  # relaxation parameters
	print("Simulation running...")
	for kk in range(pr["steps"]):
		T = g.sum(axis=0) 		# calculation of temperature
		f_2l = np.copy(f_l) 	# get a copy of liquid fraction
		for i in range(pr["lattice_y"]): # calculation of enthalpy
			for j in range(pr["lattice_x"]): #
				H_k[i][j] = pr["Cps"] * T[i][j] + f_l[i][j] * La
				if (H_k[i][j] <= H_s):
					f_l[i][j]=0.0
				elif (H_k[i][j] > H_s and H_k[i][j] < H_l):
					f_l[i][j] = (H_k[i][j] - H_s)/(H_l - H_s)
				else:
					f_l[i][j]=1.0

		# (3) transformación de la función de distribución espacio de vel a espacio momentos
		n = np.tensordot(N, g, axes=([1],[0]))

		# (4) colision step
		for i in range(pr["lattice_y"]):
			for j in range(pr["lattice_x"]):
				alpha_e[i][j] = alpha_l*f_l[i][j] + pr["alpha_s"]*(1.0 - f_l[i][j])
				tau_t[i][j] = 0.5 + alpha_e[i][j]/(pr["sigma"]*c_st**2.0 * pr["delta_t"])

		for i in range(pr["lattice_y"]):
			for j in range(pr["lattice_x"]):
				relax_o = np.array([[1.0, 0.0, 0.0, 0.0, 0.0],[0.0, 1.0/(tau_t[i][j]), 0.0, 0.0, 0.0],
									[0.0, 0.0, 1.0/(tau_t[i][j]), 0.0, 0.0],[0.0, 0.0, 0.0, 1.5, 0.0],
									[0.0, 0.0, 0.0, 0.0, 1.5]])  # Relaxation matrix in MRT
				s_surce = np.array([-((pr["poros"] * La)/(pr["sigma"]*pr["Cpl"]))*(f_l[i][j]-f_2l[i][j])/1.0, 0.0, 0.0,
									-pr["w_test"]*((pr["poros"]*La)/(pr["sigma"]*pr["Cpl"]))*(f_l[i][j]-f_2l[i][j])/1.0, 0.0])
				n_eq = np.array([T[i][j], 0.0, 0.0, pr["w_test"]*T[i][j] , 0.0]) # equilibrium distribution function
				for k in range(5):
					n_res1[k] = n[k][i][j] - n_eq[k]
				n_res2 = np.dot(relax_o, n_res1)
				for k in range(5):
					n[k][i][j] = n[k][i][j] - n_res2[k] + s_surce[k]  # main colisión

		# (5) Contain temperature and liquid fraction
		g = np.tensordot(N_inv, n, axes=([1], [0]))  # linear transformation from moment space to velocity space

		# Propagation step
		for i in range(pr["lattice_y"]):
			for j in range(pr["lattice_x"]-1,0,-1):  # propagate horizontal distribution functions (df)
				g[1][i][j] = g[1][i][j-1]  # df 1
			for	j in range(pr["lattice_x"]-1):
				g[3][i][j] = g[3][i][j+1]  # df 3

		for i in range(pr["lattice_y"]-1):  # propagate vertical distribution functions
			for j in range(pr["lattice_x"]):
				g[2][i][j] = g[2][i+1][j]  # df 2

		for i in range(pr["lattice_y"] - 1, 0, -1):
			for j in range(pr["lattice_x"]):
				g[4][i][j] = g[4][i-1][j] # df 4

		# () boundary conditions
		for i in range(pr["lattice_y"]):
			g[1][i][0] = w_s[1]*pr["T_b"] + w_s[3] * pr["T_b"] - g[3][i][0]
			g[3][i][pr["lattice_x"]-1] = w_s[1]*pr["T_i"] + w_s[3] * pr["T_i"] - g[1][i][pr["lattice_x"] - 1]

		for j in range(pr["lattice_x"]):  # Periodic horizontal boundary conditions
			g[2][pr["lattice_y"]-1][j] = g[2][0][j]
			g[4][0][j] = g[4][pr["lattice_y"]-1][j]

	logger.debug("Simulation finished")
	t = time.time() - tiempo_inicio

	#print"\n", t, lattice_x*lattice_y*steps/t/1e6
	Perfo = [t, pr["lattice_x"] * pr["lattice_y"] * pr["steps"]/t/1e6]  # simulation time, MLUPS
	print ("time: ",t, " [s]")
	#print "N: ",N
	#print "N_inv: ",N_inv
	#print N.shape
	#print g.shape
	T = g.sum(axis=0)
	print ("Temperature= ", T)

	logger.debug("save results in txt format")
	np.savetxt('Results/Performance_.txt',Perfo) # Save performance
	np.savetxt('Results/T_.txt', T,fmt='%.6f'  ) # Save temperature
	np.savetxt('Results/f_l_.txt', f_l,fmt='%.3f') # Save liquid fraction
