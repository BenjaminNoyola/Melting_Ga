#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Authors: Benjamín S. Noyola García, Suemi Rogriguez Romo
# 
# Numerical Solution 1D performed by MRT-LB, Lattice of 254*8, 
# the phase change of a bar is modeled in 1D section 4.1 of 
# paper: Qing, Liu, Ya-Ling He*. Double multiple-relaxation-time 
# lattice Boltzmann Model for solid-liquid phase change with natural 
# convection in porous media, Physica A. 438(2015) 94-106.
#
# Performed using Python-Numpy


import d2q9_nxnyns as d2q9
from numba import cuda, float64, float32
import numpy as np
import matplotlib.pyplot as plt

getg = cuda.jit('void (f8[:,:,:], f8[::1], i8, i8)', device=True)(d2q9.getg)
getfl = cuda.jit('void (f8[:,:], f8[:1], i8, i8)', device=True)(d2q9.getfl)
calc_T = cuda.jit('void (f8[::1], f8[:1])', device=True)(d2q9.calc_T)
calc_copiafl = cuda.jit('void (f8[:1],f8[:1])', device=True)(d2q9.calc_copiafl)
calc_Hk = cuda.jit('void (f8[:1], f8[:1], f8[:1])', device=True)(d2q9.calc_Hk)
calc_fl = cuda.jit('void (f8[:1], f8[:1])', device=True)(d2q9.calc_fl)
calc_g2n = cuda.jit('void (f8[::1], f8[::1])', device=True)(d2q9.calc_g2n)

calc_alfe = cuda.jit('void (f8[:1], f8[:1], f8[1])', device=True)(d2q9.calc_alfe)
calc_taut = cuda.jit('void (f8[:1], f8[:1])', device=True)(d2q9.calc_taut)
calc_relax = cuda.jit('void (f8[::1], f8[:1])', device=True)(d2q9.calc_relax)
calc_Ssurce = cuda.jit('void (f8[::1], f8[:1], f8[:1])', device=True)(d2q9.calc_Ssurce)
n_eq_loc = cuda.jit('void (f8[::1], f8[:1])', device=True)(d2q9.n_eq_loc)
calc_colision = cuda.jit('void (f8[::1], f8[::1], f8[::1], f8[::1])', device=True)(d2q9.calc_colision)
n2g_loc = cuda.jit('void (f8[::1], f8[::1])', device=True)(d2q9.n2g_loc)
setfl = cuda.jit('void (f8[:,:], f8[:1], i8, i8)', device=True)(d2q9.setfl)
setg = cuda.jit('void (f8[:,:,:], f8[::1], i8, i8)', device=True)(d2q9.setg)
set_prueba = cuda.jit('void (f8[:,:], f8[:1], i8, i8)', device=True)(d2q9.set_prueba)


@cuda.jit('void(f8[:,:,:],f8[:,:,:])')
def propagacion(d_g, copia_g):
    nx, ny, ns = d_g.shape
    i, j = cuda.grid(2)
    
    copia_g[i,j,1] = d_g[i,j,1]
    copia_g[i,j,2] = d_g[i,j,2]
    copia_g[i,j,3] = d_g[i,j,3]
    copia_g[i,j,4] = d_g[i,j,4]
    

    for i in xrange (ny):
        for j in xrange (nx-1,0,-1): # propagation of horizontal elements
            d_g[j][i][1] = copia_g[j-1][i][1] #vector 1
        for	j in xrange (nx-1):
            d_g[j][i][3] = copia_g[j+1][i][3] #vector 3
    	
    for i in xrange (ny-1): #propagation of vertical elements
        for j in xrange (nx): 
            d_g[j][i][2] = copia_g[j][i+1][2] #vector 2
    	
    for i in xrange (ny-1,0,-1): 
        for j in xrange (nx):  
            d_g[j][i][4] = copia_g[j][i-1][4] #vector 4
    


@cuda.jit('void(f8[:,:,:], f8[:], f8[:], f8[:])')
def condiciones_frontera(d_g, d_ws, d_Tb, d_Ti):
    nx, ny, ns = d_g.shape
#    i, j = cuda.grid(2)
    
    for i in xrange(ny): # boundary condition in east and west
        d_g[0,i,1] = d_ws[1]*d_Tb[0] + d_ws[3]*d_Tb[0] - d_g[0,i,3]
        d_g[nx-1][i][3] = d_ws[1]*d_Ti[0] + d_ws[3]*d_Ti[0] - d_g[nx-1][i][1]
		
    for j in xrange(nx): # boundary condition in north and south
        d_g[j][ny-1][2] = d_g[j][0][2]
        d_g[j][0][4] = d_g[j][ny-1][4]
    
       
@cuda.jit('void(f8[:,:,:], f8[:,:], f8[:])')
def one_time_step(d_g, d_fl, D_al): 
    nx, ny, ns = d_g.shape 
    gloc = cuda.local.array(5, dtype=float64) 
    nloc = cuda.local.array(5, dtype=float64)      
    neqloc = cuda.local.array(5, dtype=float64)
    Tloc = cuda.local.array(1, dtype=float64)
    Hkloc = cuda.local.array(1, dtype=float64)    
    flloc = cuda.local.array(1, dtype=float64)
    copiaflloc = cuda.local.array(1, dtype=float64)
    alfeloc= cuda.local.array(1, dtype=float64)
    tautloc = cuda.local.array(1, dtype=float64)
    relaxloc= cuda.local.array(5, dtype=float64)
    Ssurceloc=cuda.local.array(5, dtype=float64)
    
    i, j = cuda.grid(2)  

    getg(d_g, gloc, i, j)
    getfl(d_fl, flloc, i ,j)
    calc_T(gloc, Tloc)           # calculation of temperature 
    calc_copiafl(copiaflloc, flloc)
    calc_Hk(Tloc, Hkloc, flloc)  # enthalpy calculation
    calc_fl(flloc, Hkloc)        # calculate liquid fraction
    calc_g2n(nloc, gloc)         # linear transformation from velocity space to moment space
    calc_alfe(alfeloc, flloc, D_al)
    calc_taut(tautloc, alfeloc)    
    calc_relax(relaxloc, tautloc)    
    calc_Ssurce(Ssurceloc, flloc, copiaflloc)
    n_eq_loc(neqloc, Tloc)
    calc_colision(nloc, relaxloc, neqloc, Ssurceloc)    
    n2g_loc(gloc, nloc)
    
    setfl(d_fl, flloc, i, j)
    setg(d_g, gloc, i, j)
