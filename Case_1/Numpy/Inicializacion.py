####################################
# Parameters of the case 1 code
####################################
import numpy as np
import get_parser_parameters as gp
from functools import lru_cache
pr = gp.get_parameters()
# import time

#############################################
# Initial conditions
#############################################
@lru_cache
def weights():
    w_s = np.zeros([5])  # weight coefficients
    # weights D2Q5:
    w_s[0] = (1 - pr["w_test"]) / 5.
    for i in range(1, 5):
        w_s[i] = (4 + pr["w_test"]) / 20.
    return w_s

@lru_cache
def fl():
    # initial conditions of temperature and liquid fraction
    f_l = np.zeros([pr["lattice_y"], pr["lattice_x"]])  # liquid fraction is saved in a matrix
    for i in range(pr["lattice_y"]):
        f_l[i][0] = 1.0
        f_l[i][pr["lattice_x"] - 1] = 0.0
    return f_l

@lru_cache
def T_():
    # initial conditions of temperature and liquid fraction
    T = np.ones([pr["lattice_y"], pr["lattice_x"]])  # Temperature is saved in a matrix
    T = pr["T_i"] * T
    for i in range(pr["lattice_y"]):
        T[i][0] = pr["T_b"]
        T[i][pr["lattice_x"] - 1] = pr["T_i"]
    return T

@lru_cache
def enthalpy():
    T, f_l = T_(), fl()
    La = pr["Cpl"] * (pr["T_b"] - pr["T_m"]) / pr["St"]  # latent heat
    # calculation of enthalpy at the beginning
    H_k = np.zeros([pr["lattice_y"], pr["lattice_x"]])  # enthalpy is saved in a matrix
    for i in range(pr["lattice_y"]):
        for j in range(pr["lattice_x"]):
            H_k[i][j] = pr["Cps"] * T[i][j] + f_l[i][j] * La
    return H_k

@lru_cache
def distribution_func():
    g = np.zeros([5, pr["lattice_y"], pr["lattice_x"]])  # distribution function is saved in a tensor order 3
    # initial distribution function
    w_s, T = weights(), T_()
    for i in range(pr["lattice_y"]):
        for j in range(pr["lattice_x"]):
            for k in range(5):
                g[k][i][j] = w_s[k] * T[i][j]
    return g

@lru_cache
def N_():
    N = np.array([[1.0, 1.0, 1.0, 1.0, 1.0], [0.0, 1.0, 0.0, -1.0, 0.0],
                  [0.0, 0.0, 1.0, 0.0, -1.0], [-4.0, 1.0, 1.0, 1.0, 1.0],
                  [0.0, 1.0, -1.0, 1.0, -1.0]])
    return N

@lru_cache
def N_inv():
    N = N_()
    return np.linalg.inv(N)

# print("pruebas: ", pr["lattice_x"])
# uno = time.time()
# distribution_func()
# print(time.time()-uno)
# dos = time.time()
# distribution_func()
# print(time.time()-dos)
