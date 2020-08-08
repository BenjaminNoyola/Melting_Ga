
# Analytic solution of a conduction melting in a semi-infinite space
# and its comparison with numeric solution. The problem is defined in
# paper: Simulations of Ga melting based on multiple-relaxation time
# lattice Boltzmann method performed with CUDA in Python, Suemi Rodriguez Romo,
# Benjamin Noyola, 2020

# Authors: Benjamín S. Noyola García, Suemi Rodriguez Romo

# Plot Analytic Vs numeric solution
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from scipy.special import erf
from scipy.optimize import brentq
import get_parser_parameters as gp
import logging, sys
np.seterr(divide='ignore')

if __name__=='__main__':

	logger = logging.getLogger('LOG')
	logger.setLevel(logging.DEBUG)
	fh = logging.FileHandler('LOGS_SolAnVsNum.log')
	fh.setLevel(logging.DEBUG)
	logger.addHandler(fh)
	formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	fh.setFormatter(formatter)
	logger.addHandler(fh)



	pr = gp.get_parameters()

	##################################################################
	# Load txt file of temperature
	##################################################################
	logger.debug("Loading txt file of temperature")
	try:
		matriz = np.loadtxt('T_.txt')
	except Exception as e:
		logger.critical("No se pudo cargar el archico: ")
		sys.exit()
	size = pr["lattice_x"] # size
	ydata = np.zeros([size])
	for i in range(size):
		ydata[i] = matriz[int(pr["lattice_y"]//2)][i]
	xdata = np.linspace(0, size-1, size)

	##################################################################
	# calculate parameters
	###################################################################
	logger.debug("Calcutation of initial parameters")
	alpha_l= pr["Delta_alfa"] * pr["alpha_s"]  # thermal diffusivity of liquid
	k_s= pr["alpha_s"] * pr["rho"] * pr["Cps"]  # solid thermal conductivity
	k_l= alpha_l * pr["rho"] * pr["Cpl"]  # liquid thermal conductivity
	La= pr["Cpl"] * (pr["T_b"] - pr["T_m"]) / pr["St"]  # latent heat
	H_l= pr["Cpl"] * pr["T_l"] + 1.0 * La  # Enthalpy of liquid  0.04/2 = 0.02,  T_m=0 -> Tl=0.02; fl=1
	H_s= pr["Cps"] * pr["T_s"] + 0.0 * La  # Enthalpy of solid  Ts=-0.02, fl=0
	t= (pr["F_0"] * pr["H"] ** 2) / pr["alpha_s"]  # time step
	c= pr["delta_x"] / pr["delta_t"]  # velocidad de lattice

	t = (pr["F_0"]*pr["H"]**2)/pr["alpha_s"]
	t = pr["steps"]  # Edit this line to be the same than the MRT-LB program
	print("time: ", t)

	##################################################################
	#Analitic solution of a no linear equation
	###################################################################
	logger.debug("Starting analytical solution")
	def f(n):
		return (np.exp(-n**2))/(erf(n)) + (k_s/k_l) * np.sqrt(alpha_l/pr["alpha_s"])* \
			   ((pr["T_i"] - pr["T_m"])/(pr["T_b"] - pr["T_m"])) * (np.exp(-n**2 * (alpha_l/pr["alpha_s"]))/ \
		erfc(n*np.sqrt(alpha_l/pr["alpha_s"]))) - ((n*La * np.sqrt(np.pi))/(pr["Cpl"]*(pr["T_b"] - pr["T_m"])))

	n = brentq(f, 0.0, 5.0) #  brentq function, find a root in interval [0, 5.0]
	xm = 2.0 * n * np.sqrt(alpha_l * t)
	# print("xm = ", xm, "\n")
	x = np.linspace(0, size-1, size)
	dominio_1 = np.linspace(0,1,size)
	dominio_2 = np.linspace(0,1,size)
	T = np.zeros(size)
	tetha = np.zeros(size)

	for i in range (size):
		if i <= (size/size)*(xm):
			T[i] = pr["T_b"] - ((pr["T_b"] - pr["T_m"]) * erf(x[i]/(2. * np.sqrt(alpha_l*t))))/erf(n)
			tetha[i] = (T[i]-pr["T_b"])/(pr["T_b"] - pr["T_m"])
		else:
			T[i]=pr["T_i"] + ((pr["T_m"] - pr["T_i"])*erfc(x[i]/(2.*np.sqrt(pr["alpha_s"]*t))))/\
				 erfc(n*np.sqrt(alpha_l/pr["alpha_s"]))
			tetha[i] = (T[i] - pr["T_b"])/(pr["T_b"] - pr["T_m"])

	######################################################################
	# Plot Analytic solution vs numeric solution
	######################################################################
	logger.debug("Plot analytical solution")
	analitica,=plt.plot(dominio_1, T, 'xr',lw=3, label="Analytic")
	numerica,=plt.plot(dominio_2 , ydata,'*',lw=1, label="MRT-LB")
	plt.title('Analytic vs numeric solutions')
	plt.grid(True)
	plt.grid(color='g', alpha=0.5, linestyle='solid', linewidth=1.0) # intentar
	plt.ylabel('T')
	plt.text(0.4, 0.0, r"$\alpha_l / \alpha_s = {}$".format(pr["Delta_alfa"]), fontsize=30, color="blue")
	plt.text(0.4, 0.2, r" t={}".format(pr["steps"]), fontsize=30, color="blue")
	plt.ylim(-1.2, 1)
	plt.xlabel('x/H')
	plt.legend()
	plt.savefig("AnVsNum.png")
	plt.show()