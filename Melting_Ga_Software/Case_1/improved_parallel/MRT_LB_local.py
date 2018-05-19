#! /usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from numba import cuda

def getg(d_g, gloc, i, j):
    gloc[0] = d_g[i, j, 0]
    gloc[1] = d_g[i, j, 1]
    gloc[2] = d_g[i, j, 2]
    gloc[3] = d_g[i, j, 3]
    gloc[4] = d_g[i, j, 4]

def getfl(d_fl, flloc, i, j):
    flloc[0] = d_fl[i,j]


def calc_T(gloc, Tloc):
    Tloc[0] = gloc[0] + gloc[1] + gloc[2] + gloc[3] + gloc[4]

def calc_copiafl(copiaflloc, flloc):
    copiaflloc[0] = flloc[0]    
        
    
def calc_Hk(Tloc, Hkloc, flloc):
    Cps = 1.0   #calor especifico del solido y calor latente
    La = 1.0   
    Hkloc[0] = Cps*Tloc[0] + flloc[0]*La #Tloc[0]#d_fl[i,j] 
#    Hkloc[0] = 1.0+fllloc[0] 
    #Hkloc[0] = Hkloc[0]+Tloc[0]#d_fl[i,j]


def calc_fl(flloc, Hkloc): #calculo de la entalpia
    Hs = -0.02   #H_s = Cps*(-0.02) + 0.0*La , H_l = Cpl*0.02 + 1.0*La  
    Hl = 1.02       
    if (Hkloc[0] <= Hs):
        flloc[0] = 0.0
    elif ((Hkloc[0] > Hs) and (Hkloc[0] < Hl)):
        flloc[0] = (Hkloc[0] - Hs)/(Hl - Hs)
#        flloc[0] = (Hkloc[0] + 0.02)/(1.02 + 0.02)
    else:
        flloc[0] = 1.0
     
def calc_g2n(nloc, gloc): #transformación lineal
    nloc[0] = gloc[0] + gloc[1] + gloc[2] + gloc[3] + gloc[4]    
    nloc[1] = gloc[1] - gloc[3]    
    nloc[2] = gloc[2] - gloc[4]
    nloc[3] = -4.0*gloc[0] + gloc[1] + gloc[2] + gloc[3] + gloc[4] 
    nloc[4] = gloc[1] - gloc[2] + gloc[3] - gloc[4]

def calc_alfe(alfeloc, flloc, D_al):    
    alfa_s = 0.002   
    alfa_l = D_al[0]*alfa_s 
    alfeloc[0] = alfa_l*flloc[0] + alfa_s*(1.0-flloc[0])

def calc_taut(tautloc, alfeloc):
    sigma=1.0
    c_st =0.4472135955
    delta_t=1.0   
    tautloc[0] = 0.5 + alfeloc[0]/(sigma*c_st**2*delta_t)
    
def calc_relax(relaxloc, tautloc):
    relaxloc[0] = 1.0
    relaxloc[1] = 1.0/tautloc[0]
    relaxloc[2] = 1.0/tautloc[0]    
    relaxloc[3] = 1.5
    relaxloc[4] = 1.5



def calc_Ssurce(Ssurceloc, flloc, copiaflloc):
    poros, La, sigma, Cpl, w_test = 1.00, 1.00, 1.00, 1.00, -2.00
    Ssurceloc[0] = -((poros*La)/(sigma*Cpl))*(flloc[0] - copiaflloc[0])/1.0
    Ssurceloc[1] = 0.0
    Ssurceloc[2] = 0.0
    Ssurceloc[3] = -w_test*((poros*La)/(sigma*Cpl))*(flloc[0] - copiaflloc[0])/1.0 
    Ssurceloc[4] = 0.0   


def n_eq_loc(neqloc, Tloc):
    w_test = -2.0    
    neqloc[0] = Tloc[0]
    neqloc[1] = 0.0
    neqloc[2] = 0.0
    neqloc[3] = w_test*Tloc[0]  
    neqloc[4] = 0.0

def calc_colision(nloc, relaxloc, neqloc, Ssurceloc):   
    nloc[0] = nloc[0] - relaxloc[0] * (nloc[0]-neqloc[0]) + Ssurceloc[0]
    nloc[1] = nloc[1] - relaxloc[1] * (nloc[1]-neqloc[1]) + Ssurceloc[1]
    nloc[2] = nloc[2] - relaxloc[2] * (nloc[2]-neqloc[2]) + Ssurceloc[2]
    nloc[3] = nloc[3] - relaxloc[3] * (nloc[3]-neqloc[3]) + Ssurceloc[3]
    nloc[4] = nloc[4] - relaxloc[4] * (nloc[4]-neqloc[4]) + Ssurceloc[4]

def n2g_loc(gloc, nloc):
    gloc[0] = 0.2*nloc[0] - 0.2*nloc[3]
    gloc[1] = 0.2*nloc[0] + 0.5*nloc[1] + 0.05*nloc[3] + 0.25*nloc[4]
    gloc[2] = 0.2*nloc[0] + 0.5*nloc[2] + 0.05*nloc[3] - 0.25*nloc[4]    
    gloc[3] = 0.2*nloc[0] - 0.5*nloc[1] + 0.05*nloc[3] + 0.25*nloc[4]
    gloc[4] = 0.2*nloc[0] - 0.5*nloc[2] + 0.05*nloc[3] - 0.25*nloc[4]


def setfl(d_fl, flloc, i, j):
    d_fl[i,j] = flloc[0]

#def setT(d_T, Tloc, i, j):
#    d_T[i,j] = Tloc[0]

def setg(d_g, gloc, i, j):
    d_g[i, j, 0] = gloc[0]
    d_g[i, j, 1] = gloc[1]
    d_g[i, j, 2] = gloc[2]
    d_g[i, j, 3] = gloc[3]
    d_g[i, j, 4] = gloc[4]

def set_prueba(d_fl, flloc, i, j):
    d_fl[i,j] = flloc[0]