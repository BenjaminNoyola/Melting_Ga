#! /usr/bin/env python
# -*- coding: utf-8 -*-
import MRT_LB_local as d2q5
from numba import cuda, float64, float32
import numpy as np
import matplotlib.pyplot as plt

getg = cuda.jit('void (f8[:,:,:], f8[::1], i8, i8)', device=True)(d2q5.getg)
getfl = cuda.jit('void (f8[:,:], f8[:1], i8, i8)', device=True)(d2q5.getfl)
#getn = cuda.jit('void (f8[:,:,:], f8[::1], i8, i8)', device=True)(d2q5.getn)
calc_T = cuda.jit('void (f8[::1], f8[:1])', device=True)(d2q5.calc_T)
calc_copiafl = cuda.jit('void (f8[:1],f8[:1])', device=True)(d2q5.calc_copiafl)
calc_Hk = cuda.jit('void (f8[:1], f8[:1], f8[:1])', device=True)(d2q5.calc_Hk)
calc_fl = cuda.jit('void (f8[:1], f8[:1])', device=True)(d2q5.calc_fl)
calc_g2n = cuda.jit('void (f8[::1], f8[::1])', device=True)(d2q5.calc_g2n)

calc_alfe = cuda.jit('void (f8[:1], f8[:1], f8[1])', device=True)(d2q5.calc_alfe)
calc_taut = cuda.jit('void (f8[:1], f8[:1])', device=True)(d2q5.calc_taut)
calc_relax = cuda.jit('void (f8[::1], f8[:1])', device=True)(d2q5.calc_relax)
calc_Ssurce = cuda.jit('void (f8[::1], f8[:1], f8[:1])', device=True)(d2q5.calc_Ssurce)
n_eq_loc = cuda.jit('void (f8[::1], f8[:1])', device=True)(d2q5.n_eq_loc)
calc_colision = cuda.jit('void (f8[::1], f8[::1], f8[::1], f8[::1])', device=True)(d2q5.calc_colision)
n2g_loc = cuda.jit('void (f8[::1], f8[::1])', device=True)(d2q5.n2g_loc)

setfl = cuda.jit('void (f8[:,:], f8[:1], i8, i8)', device=True)(d2q5.setfl)
#setT = cuda.jit('void (f8[:,:], f8[:1], i4, i4)', device=True)(d2q5.setT)
setg = cuda.jit('void (f8[:,:,:], f8[::1], i8, i8)', device=True)(d2q5.setg)
set_prueba = cuda.jit('void (f8[:,:], f8[:1], i8, i8)', device=True)(d2q5.set_prueba)


@cuda.jit('void(f8[:,:,:],f8[:,:,:])')
def propagacion(d_g, copia_g):
    nx, ny, ns = d_g.shape
    i, j = cuda.grid(2)
    
    copia_g[i,j,1] = d_g[i,j,1]
    copia_g[i,j,2] = d_g[i,j,2]
    copia_g[i,j,3] = d_g[i,j,3]
    copia_g[i,j,4] = d_g[i,j,4]
        
    ### el interior sin el perimetro ###
    if i>0 and i<nx-1 and j>0 and j<ny-1:
        d_g[i,j,1] = copia_g[i-1,j,1]        
        d_g[i,j,2] = copia_g[i,j+1,2]
        d_g[i,j,3] = copia_g[i+1,j,3]
        d_g[i,j,4] = copia_g[i,j-1,4]

    ### el perimetro sin esquinas ###
    if j>0 and j<ny-1:
        d_g[nx-1,j,1] = copia_g[nx-2,j,1] 
        d_g[nx-1,j,2] = copia_g[nx-1,j+1,2]
        d_g[nx-1,j,4] = copia_g[nx-1,j-1,4]

    if i>0 and i<nx-1:
        d_g[i,0,2] = copia_g[i,1,2]
        d_g[i,0,1] = copia_g[i-1,0,1]        
        d_g[i,0,3] = copia_g[i+1,0,3]

    if j>0 and j<ny-1:
        d_g[0,j,3] = copia_g[1,j,3] 
        d_g[0,j,2] = copia_g[0,j+1,2] 
        d_g[0,j,4] = copia_g[0,j-1,4]

    if i>0 and i<nx-1:
        d_g[i,ny-1,4] = copia_g[i,ny-2,4]
        d_g[i,ny-1,1] = copia_g[i-1,ny-1,1]   
        d_g[i,ny-1,3] = copia_g[i+1,ny-1,3]
  
    ### las esquinas ###    
    d_g[nx-1,0,1] = copia_g[nx-2,0,1]        
    d_g[nx-1,0,2] = copia_g[nx-1,1,2]        
    
    d_g[0,0,2] = copia_g[0,1,2]
    d_g[0,0,3] = copia_g[1,0,3]

    d_g[0,ny-1,3] = copia_g[1,ny-1,3]
    d_g[0,ny-1,4] = copia_g[0,ny-2,4]
   
    d_g[nx-1,ny-1,1] = copia_g[nx-2,ny-1,1]         
    d_g[nx-1,ny-1,4] = copia_g[nx-1,ny-2,4]
        
        
#@cuda.jit('void(f8[:,:,:], f8[:], f8[:], f8[:])')
#def condiciones_frontera(d_g, d_ws, d_Tb, d_Ti):
@cuda.jit('void(f8[:,:,:], f8[:])')
def condiciones_frontera(d_g, d_ws): 
    nx, ny, ns = d_g.shape
#    i, j = cuda.grid(2)
    d_Tb = 1.0
    d_Ti = -1.0    
#    d_ws = [0.6, 0.1, 0.1, 0.1, 0.1]
#    copia_g[i,j,1] = d_g[i,j,1]
#    copia_g[i,j,2] = d_g[i,j,2]
#    copia_g[i,j,3] = d_g[i,j,3]
#    copia_g[i,j,4] = d_g[i,j,4]
    
    for i in xrange(ny):
        d_g[0,i,1] = d_ws[1]*d_Tb + d_ws[3]*d_Tb - d_g[0,i,3]
        d_g[nx-1,i,3] = d_ws[1]*d_Ti + d_ws[3]*d_Ti - d_g[nx-1,i,1]
		
    for j in xrange(nx): #condiciones de frontera periodicas horizontales
        d_g[j,ny-1,2] = d_g[j,0,2]
        d_g[j,0,4] = d_g[j,ny-1,4]
    

# ###condiciones de frontera PERIODICAS horizontales###  
#
#    if i>0 and i<nx-1:
#        d_g[i, 0, 4] = copia_g[i, ny-1, 4]        
#        d_g[i, ny-1, 2] = copia_g[i, 0, 2]
#        
#    d_g[0,0,4]=copia_g[0,ny-1,4]    
#    d_g[nx-1,0,4]=copia_g[nx-1,ny-1,4]
#    d_g[0,ny-1,2]=copia_g[0,0,2]
#    d_g[nx-1,ny-1, 2]=copia_g[nx-1,0,2]  
#
####condiciones de frontera de DIRICHLET verticales###
#
#    if j>0 and j<ny-1:
#        d_g[0, j, 1] = d_ws[1]*d_Tb[0] + d_ws[3]*d_Tb[0] - copia_g[0, j, 3]
#        d_g[nx-1, j, 3] = d_ws[1]*d_Ti[0] + d_ws[3]*d_Ti[0] - copia_g[nx-1, j, 1]
#       
#    d_g[0, 0, 1] = d_ws[1]*d_Tb[0] + d_ws[3]*d_Tb[0] - copia_g[0,0,3]
#    d_g[nx-1, 0, 3] = d_ws[1]*d_Ti[0] + d_ws[3]*d_Ti[0] - copia_g[nx-1, 0, 1]
#    d_g[0, ny-1, 1] = d_ws[1]*d_Tb[0] + d_ws[3]*d_Tb[0] - copia_g[0,ny-1,3]
#    d_g[nx-1, ny-1, 3] = d_ws[1]*d_Ti[0] + d_ws[3]*d_Ti[0] - copia_g[nx-1, ny-1, 1]

    
 
       
@cuda.jit('void(f8[:,:,:], f8[:,:], f8[:])')
def collision_local(d_g, d_fl, D_al): 
    nx, ny, ns = d_g.shape #se asignan valores 
    gloc = cuda.local.array(5, dtype=float64)  # crea una lista de manera local  
    nloc = cuda.local.array(5, dtype=float64)      
    neqloc = cuda.local.array(5, dtype=float64)
    Tloc = cuda.local.array(1, dtype=float64)
    Hkloc = cuda.local.array(1, dtype=float64)    
    flloc = cuda.local.array(1, dtype=float64)  ##
    copiaflloc = cuda.local.array(1, dtype=float64)
    alfeloc= cuda.local.array(1, dtype=float64)
    tautloc = cuda.local.array(1, dtype=float64)
    relaxloc= cuda.local.array(5, dtype=float64)
    Ssurceloc=cuda.local.array(5, dtype=float64)
    
    i, j = cuda.grid(2)  #Define a los contadores(identificadores) que corren sobre la red cuda

    getg(d_g, gloc, i, j)           #se obtiene g de manera local
    getfl(d_fl, flloc, i ,j)
#    getn(d_n, nloc, i, j)           #se obtiene g de manera local
    calc_T(gloc, Tloc)         #cálculo de la temperatura
    calc_copiafl(copiaflloc, flloc)
    calc_Hk(Tloc, Hkloc, flloc)  #cálculo de la entalpia
#    cuda.syncthreads()    
    calc_fl(flloc, Hkloc) #cálculo de la fracción de líquido
#    cuda.syncthreads()    
    calc_g2n(nloc, gloc) #transformación lineal del espacio de velocidades al espacio de momentos
    calc_alfe(alfeloc, flloc, D_al)#cálculo de alfa_e 
    calc_taut(tautloc, alfeloc)    
    calc_relax(relaxloc, tautloc)    
    calc_Ssurce(Ssurceloc, flloc, copiaflloc)
    n_eq_loc(neqloc, Tloc)
    calc_colision(nloc, relaxloc, neqloc, Ssurceloc)    
    n2g_loc(gloc, nloc)
    
#    cuda.syncthreads()
    setfl(d_fl, flloc, i, j)
    setg(d_g, gloc, i, j)
#    cuda.syncthreads()
#    set_prueba(d_T, Tloc, i, j)
