#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Authors: Benjamín S. Noyola García, Suemi Rodriguez Romo

import local as loc
from numba import cuda, float64, float32
import numpy as np
import get_parser_parameters as gp
pr = gp.get_parameters()  # pr is a dictionary of parameters
T_h = pr["T_h"]
T_c = pr["T_c"]

# links to  local.py program
###################### velocity field #######################################
getf = cuda.jit('void (f8[:,:,:], f8[::1], i8, i8)', device=True)(loc.getf) 
getux = cuda.jit('void (f8[:,:], f8[::1], i8, i8)', device=True)(loc.getux)
getuy = cuda.jit('void (f8[:,:], f8[::1], i8, i8)', device=True)(loc.getuy)
getFx = cuda.jit('void (f8[:,:], f8[::1], i8, i8)', device=True)(loc.getFx)
getFy = cuda.jit('void (f8[:,:], f8[::1], i8, i8)', device=True)(loc.getFy)
getden = cuda.jit('void (f8[:,:], f8[::1], i8, i8)', device=True)(loc.getden)
getf_l = cuda.jit('void (f8[:,:], f8[::1], i8, i8)', device=True)(loc.getf_l)
getf_2l = cuda.jit('void (f8[:,:], f8[::1], i8, i8)', device=True)(loc.getf_2l)
getT = cuda.jit('void (f8[:,:], f8[::1], i8, i8)', device=True)(loc.getT)
f2m = cuda.jit('void (f8[::1], f8[::1])', device=True)(loc.f2m)
calc_fl_PCMloc=cuda.jit('void (f8[::1], f8[::1])', device=True)(loc.calc_fl_PCMloc)
calc_noruloc=cuda.jit('void (f8[::1], f8[::1], f8[::1])', device=True)(loc.calc_noruloc)
calc_m_eqloc=cuda.jit('void (f8[::1], f8[::1], f8[::1], f8[::1], f8[::1], f8[::1], f8[::1])', device=True)(loc.calc_m_eqloc)
calc_Sloc=cuda.jit('void (f8[::1], f8[::1], f8[::1], f8[::1], f8[::1], f8[::1], f8[::1])', device=True)(loc.calc_Sloc)
operloc = cuda.jit('void (f8[::1],f8[::1], f8[::1], f8[::1], f8[::1], f8[::1])', device=True)(loc.operloc)
colision =  cuda.jit('void (f8[::1], f8[::1], f8[::1])', device=True)(loc.colision) 
m2f = cuda.jit('void (f8[::1], f8[::1])', device=True)(loc.m2f)
setf = cuda.jit('void (f8[:,:,:], f8[::1], i8, i8)', device=True)(loc.setf)
calc_denloc = cuda.jit('void (f8[::1], f8[::1])', device=True)(loc.calc_denloc)
calc_Hlsloc = cuda.jit('void (f8[::1], f8[::1])', device=True)(loc.calc_Hlsloc)
calc_cfloc = cuda.jit('void (f8[::1], f8[::1], f8[::1])', device=True)(loc.calc_cfloc)
calc_sigmaloc = cuda.jit('void (f8[::1], f8[::1])', device=True)(loc.calc_sigmaloc)
calc_tau_alpha_vl = cuda.jit('void (f8[::1], f8[::1], f8[::1], f8[::1], f8[::1], f8[::1])', device=True)(loc.calc_tau_alpha_vl)
calc_lloc = cuda.jit('void (f8[::1], f8[::1], f8[::1], f8[::1], f8[::1])', device=True)(loc.calc_lloc)
calc_Gloc = cuda.jit('void (f8[::1], f8[::1], f8[::1], f8[::1],f8[::1])', device=True)(loc.calc_Gloc)
calc_Vloc = cuda.jit('void (f8[::1], f8[::1], f8[::1], f8[::1], f8[::1], f8[::1])', device=True)(loc.calc_Vloc)
calc_Uloc = cuda.jit('void (f8[::1], f8[::1], f8[::1], f8[::1], f8[::1], f8[::1], f8[::1], f8[::1], f8[::1], i8,i8)', device=True)(loc.calc_Uloc)
calc_Floc = cuda.jit('void (f8[::1], f8[::1], f8[::1], f8[::1], f8[::1], f8[::1], f8[::1], f8[::1], f8[::1], i8, i8)', device=True)(loc.calc_Floc)
setvar2D = cuda.jit('void (f8[:,:], f8[::1], i8, i8)', device=True)(loc.setvar2D)
############################ Temperature field ##############################
getg = cuda.jit('void (f8[:,:,:], f8[::1], i8, i8)', device=True)(loc.getg)
g2n = cuda.jit('void (f8[::1], f8[::1])', device=True)(loc.g2n)
calc_neqloc = cuda.jit('void (f8[::1], f8[::1], f8[::1], f8[::1])', device=True)(loc.calc_neqloc)
calc_tautloc = cuda.jit('void (f8[::1], f8[::1], f8[::1])', device=True)(loc.calc_tautloc)
calc_relaxloc = cuda.jit('void (f8[::1], f8[::1])', device=True)(loc.calc_relaxloc)
calc_Ssurceloc = cuda.jit('void (f8[::1], f8[::1], f8[::1], f8[::1])', device=True)(loc.calc_Ssurceloc)
colis_g = cuda.jit('void (f8[::1], f8[::1], f8[::1], f8[::1])', device=True)(loc.colis_g)
n2g = cuda.jit('void (f8[::1], f8[::1])', device=True)(loc.n2g)
setg = cuda.jit('void (f8[:,:,:], f8[::1], i8, i8)', device=True)(loc.setg)
calc_Tloc = cuda.jit('void (f8[::1], f8[::1])', device=True)(loc.calc_Tloc)
calc_cpfl = cuda.jit('void (f8[::1], f8[::1])', device=True)(loc.calc_cpfl)
calc_Hk = cuda.jit('void (f8[::1], f8[::1], f8[::1])', device=True)(loc.calc_Hk)
calc_fl = cuda.jit('void (f8[::1], f8[::1])', device=True)(loc.calc_fl)

###############################################################################
#################### propagation step, velocity field ########################
###############################################################################
@cuda.jit('void(f8[:,:,:], f8[:,:,:])')
def propagacion(f, f2):
    nx, ny, ns = f.shape
    i, j = cuda.grid(2)
    
    for k in range(ns):
        f2[i,j,k] = f[i,j,k]
    
    if i > 0:
        f[i,j,1] = f2[i-1,j,1]
    if j < ny-1:
        f[i,j,2] = f2[i,j+1,2]
    if i < nx-1:
        f[i,j,3] = f2[i+1,j,3]
    if j > 0:
        f[i,j,4] = f2[i,j-1,4]
    if i > 0 and j < ny-1:
        f[i,j,5] = f2[i-1,j+1,5]
    if i < nx-1 and j < ny-1:
        f[i,j,6] = f2[i+1,j+1,6]
    if i < nx-1 and j > 0:
        f[i,j,7] = f2[i+1,j-1,7]
    if i > 0 and j > 0:
        f[i,j,8] = f2[i-1,j-1,8]
    
###############################################################################
###################### propagation: temperature field #########################
###############################################################################
@cuda.jit('void(f8[:,:,:], f8[:,:,:])')
def propagacion_g(g, g2):
    nx, ny, ns = g.shape
    i, j = cuda.grid(2)
    
    for k in range(ns):
        g2[i, j, k] = g[i, j, k]
    
    if i > 0:
        g[i, j, 1] = g2[i-1, j, 1]
    if j < ny-1:
        g[i, j, 2] = g2[i, j+1, 2]
    if i < nx-1:
        g[i, j, 3] = g2[i+1, j, 3]
    if j > 0:
        g[i, j, 4] = g2[i, j-1, 4]
    
##############################################################################
################### Boundary condition: velocity field #######################
##############################################################################
@cuda.jit('void(f8[:,:,:])')
def c_frontera(f):
    nx, ny, ns = f.shape
    i, j = cuda.grid(2)
    
    if i==0:            # west
        f[i, j, 1] = f[i, j, 3]
        f[i, j, 5] = f[i, j, 7]
        f[i, j, 8] = f[i, j, 6]

    if i==nx-1:         # east
        f[i, j, 3] = f[i, j, 1]
        f[i, j, 6] = f[i, j, 8]
        f[i, j, 7] = f[i, j, 5]

    if j==0:            # north
        f[i, j, 4] = f[i, j, 2]
        f[i, j, 7] = f[i, j, 5]
        f[i, j, 8] = f[i, j, 6]
    
    if j==ny-1:         # south
        f[i, j, 2] = f[i, j, 4]
        f[i, j, 5] = f[i, j, 7]
        f[i, j, 6] = f[i, j, 8]
        
##############################################################################
################## Boundary condition: temperature field #####################
##############################################################################
@cuda.jit('void(f8[:,:,:])')
def c_frontera_g(g):
    nx, ny, ns = g.shape
    i, j = cuda.grid(2)

    w_s0, w_s1, w_s2, w_s3, w_s4 = 0.6, 0.1, 0.1, 0.1, 0.1
    if i==0:            # west
        g[i,j,1] = w_s1*T_h + w_s3*T_h - g[i,j,3]   
        
    if i==nx-1:         # east
        g[i,j,3] = w_s1*T_c + w_s3*T_c - g[i,j,1] 

    if j==0:            # North
        g[i,j,0] = g[i,j+1,0]
        g[i,j,1] = g[i,j+1,1]
        g[i,j,2] = g[i,j+1,2]
        g[i,j,3] = g[i,j+1,3]
        g[i,j,4] = g[i,j+1,4]  

    if j==ny-1:         # South
        g[i,j,0] = g[i,j-1,0]
        g[i,j,1] = g[i,j-1,1]
        g[i,j,2] = g[i,j-1,2]
        g[i,j,3] = g[i,j-1,3]
        g[i,j,4] = g[i,j-1,4]


###############################################################################
####################### collision: velocity field #############################
###############################################################################
@cuda.jit('void(f8[:,:,:],f8[:,:],f8[:,:],f8[:,:],f8[:,:],f8[:,:],f8[:,:])')
def momento(d_f, d_vel_ux, d_vel_uy, d_Fx, d_Fy, d_den, d_fl):  
    floc = cuda.local.array(9, dtype = float64)     
    uxloc = cuda.local.array(1, dtype = float64)    
    uyloc = cuda.local.array(1, dtype = float64)    
    Fxloc = cuda.local.array(1, dtype = float64)    
    Fyloc = cuda.local.array(1, dtype = float64)    
    denloc = cuda.local.array(1, dtype = float64)   
    f_lloc = cuda.local.array(1, dtype = float64)   
    nor_uloc=cuda.local.array(1, dtype = float64)
    mloc = cuda.local.array(9, dtype = float64)     
    fl_PCMloc = cuda.local.array(1, dtype = float64)
    m_eqloc = cuda.local.array(9, dtype = float64)     
    Sloc = cuda.local.array(9, dtype = float64)
    res1loc = cuda.local.array(9, dtype = float64)
    fuenloc = cuda.local.array(9, dtype = float64)
   
    i, j = cuda.grid(2)  
    
    getf(d_f, floc, i, j)       
    getux(d_vel_ux, uxloc, i, j)
    getuy(d_vel_uy, uyloc, i, j)
    getFx(d_Fx, Fxloc, i, j)
    getFy(d_Fy, Fyloc, i, j)
    getden(d_den, denloc, i, j)
    getf_l(d_fl, f_lloc, i, j) 
    f2m(mloc, floc)             
    calc_fl_PCMloc(f_lloc, fl_PCMloc)
    calc_noruloc(nor_uloc, uxloc, uyloc)
    calc_m_eqloc(m_eqloc, denloc, nor_uloc, fl_PCMloc, uxloc, uyloc, f_lloc)
    calc_Sloc(uxloc, uyloc, Fxloc, Fyloc, fl_PCMloc, Sloc, f_lloc)    
    operloc(res1loc, fuenloc, mloc, m_eqloc, Sloc, f_lloc)
    colision(mloc, res1loc, fuenloc)
    m2f(floc, mloc)    
    setf(d_f, floc, i, j)
    

# calculation of macroscopic variables: density, velocity, strength
@cuda.jit('void(f8[:,:,:],f8[:,:],f8[:,:],f8[:,:],f8[:,:],f8[:,:],f8[:,:],f8[:,:])')
def cal_den_u_F(d_f, d_vel_ux, d_vel_uy, d_Fx, d_Fy, d_den, d_fl, d_T):
    floc = cuda.local.array(9, dtype = float64)     # distribution function
    vxloc = cuda.local.array(1, dtype = float64)    # velocity x
    vyloc = cuda.local.array(1, dtype = float64)    # velocity y
    uxloc = cuda.local.array(1, dtype = float64)    # macroscopic velocity x
    uyloc = cuda.local.array(1, dtype = float64)    # macroscopic velocity y
    nor_vloc=cuda.local.array(1, dtype = float64)
    nor_uloc=cuda.local.array(1, dtype = float64)    
    Fxloc = cuda.local.array(1, dtype = float64)    # strength x
    Fyloc = cuda.local.array(1, dtype = float64)    # strength y
    denloc = cuda.local.array(1, dtype = float64)   # density
    f_lloc = cuda.local.array(1, dtype = float64)   # liquid fraction
    fl_PCMloc = cuda.local.array(1, dtype = float64)    
    H_lloc = cuda.local.array(1, dtype = float64)    
    H_sloc = cuda.local.array(1, dtype = float64)  
    cfloc = cuda.local.array(1, dtype = float64)
    Tloc = cuda.local.array(1, dtype = float64)
    sigmaloc = cuda.local.array(1, dtype = float64)
    tau_tloc = cuda.local.array(1, dtype = float64)
    alf_eloc = cuda.local.array(1, dtype = float64)
    alf_lloc = cuda.local.array(1, dtype = float64)
    vlloc = cuda.local.array(1, dtype = float64)
    l_0loc = cuda.local.array(1, dtype = float64)    
    l_1loc = cuda.local.array(1, dtype = float64)    
    Gloc = cuda.local.array(9, dtype = float64) 
    TFloc = cuda.local.array(9, dtype = float64)
    
    i, j = cuda.grid(2)  
    
    getf(d_f, floc, i, j)       
    getf_l(d_fl, f_lloc, i, j)
    getT(d_T, Tloc, i, j)    
    calc_Hlsloc(H_lloc, H_sloc)
    calc_denloc(denloc, floc)
    calc_fl_PCMloc(f_lloc, fl_PCMloc)
    calc_cfloc(cfloc, fl_PCMloc, f_lloc)
    calc_sigmaloc(Tloc, sigmaloc)
    calc_tau_alpha_vl(sigmaloc, tau_tloc, alf_eloc, alf_lloc, vlloc, f_lloc)
    calc_lloc(fl_PCMloc, vlloc, cfloc, l_0loc, l_1loc)
    calc_Gloc(vlloc, alf_lloc, Tloc, Gloc,f_lloc)
    calc_Vloc(vxloc, vyloc, fl_PCMloc, nor_vloc, Gloc, floc)
    calc_Uloc(l_0loc, l_1loc, vxloc, vyloc, nor_vloc, uxloc, uyloc, nor_uloc, fl_PCMloc, i, j)
    calc_Floc(fl_PCMloc, uxloc, uyloc, cfloc, vlloc, Gloc, TFloc, Fxloc, Fyloc, i, j)
    setvar2D(d_den, denloc, i, j)
    setvar2D(d_vel_ux, uxloc, i, j)
    setvar2D(d_vel_uy, uyloc, i, j)
    setvar2D(d_Fx, Fxloc, i, j)
    setvar2D(d_Fy, Fyloc, i, j)    
    
###############################################################################
#################### collition: temperature field #############################
###############################################################################

@cuda.jit('void(f8[:,:,:],f8[:,:],f8[:,:],f8[:,:],f8[:,:], f8[:,:])')
def energia(d_g, d_T, d_fl, d_vel_ux, d_vel_uy, d_f2l): 
    gloc = cuda.local.array(5, dtype = float64)     #función de distribución g
    nloc = cuda.local.array(5, dtype = float64)
    neqloc = cuda.local.array(5, dtype = float64)
    f_lloc = cuda.local.array(1, dtype = float64)
    f_2lloc = cuda.local.array(1, dtype = float64)
    Tloc = cuda.local.array(1, dtype = float64)
    uxloc = cuda.local.array(1, dtype = float64)
    uyloc = cuda.local.array(1, dtype = float64)
    tautloc = cuda.local.array(1, dtype = float64)
    relaxloc = cuda.local.array(5, dtype = float64)
    Ssurceloc = cuda.local.array(5, dtype = float64)
    
    i, j = cuda.grid(2)  
    
    getf_l(d_fl, f_lloc, i, j) 
    getf_2l(d_f2l, f_2lloc, i, j)
    getg(d_g, gloc, i, j)       
    g2n(nloc, gloc)
    getT(d_T, Tloc, i, j) 
    getux(d_vel_ux, uxloc, i, j)
    getuy(d_vel_uy, uyloc, i, j)
    calc_neqloc(Tloc, uxloc, uyloc, neqloc)
    calc_tautloc(tautloc, Tloc, f_lloc)
    calc_relaxloc(relaxloc, tautloc)
    calc_Ssurceloc(Ssurceloc, f_lloc, f_2lloc, Tloc)
    colis_g(nloc, neqloc, Ssurceloc, relaxloc)
    n2g(gloc, nloc)
    setg(d_g, gloc, i, j)

@cuda.jit('void(f8[:,:,:],f8[:,:],f8[:,:],f8[:,:])')
def cal_T_fl_H(d_g, d_T, d_fl, d_f2l):
    gloc = cuda.local.array(5, dtype = float64)
    Tloc = cuda.local.array(1, dtype = float64)
    f_lloc = cuda.local.array(1, dtype = float64)
    f_2lloc = cuda.local.array(1, dtype = float64)
    Hkloc = cuda.local.array(1, dtype = float64)
    
    i, j = cuda.grid(2)  
    
    getg(d_g, gloc, i, j)
    getf_l(d_fl, f_lloc, i, j)
    calc_Tloc(gloc, Tloc)
    calc_cpfl(f_lloc, f_2lloc)
        
    calc_Hk(Tloc, Hkloc, f_lloc)     
    calc_fl(f_lloc, Hkloc)        
    
    setvar2D(d_fl, f_lloc, i, j)
    setvar2D(d_T, Tloc, i, j)
    setvar2D(d_f2l, f_2lloc, i, j)
