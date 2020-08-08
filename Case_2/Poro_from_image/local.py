#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Authors: Benjamín S. Noyola García, Suemi Rodriguez Romo

import numpy as np
from numba import cuda
import math 

###############################################################################
###################### Collition step #########################
###############################################################################
def getf(d_f, floc, i, j):
    floc[0] = d_f[i, j, 0]
    floc[1] = d_f[i, j, 1]
    floc[2] = d_f[i, j, 2]
    floc[3] = d_f[i, j, 3]
    floc[4] = d_f[i, j, 4]
    floc[5] = d_f[i, j, 5]
    floc[6] = d_f[i, j, 6]
    floc[7] = d_f[i, j, 7]
    floc[8] = d_f[i, j, 8]
    
def getux(d_vel_ux, uxloc, i, j):
    uxloc[0] = d_vel_ux[i,j]   
    
def getuy(d_vel_uy, uyloc, i, j):
    uyloc[0] = d_vel_uy[i,j]   
    
def getFx(d_Fx, Fxloc, i, j):
    Fxloc[0] = d_Fx[i,j]

def getFy(d_Fy, Fyloc, i, j):
    Fyloc[0] = d_Fy[i,j]   

def getden(d_den, denloc, i, j):
    denloc[0] = d_den[i,j]   

def getf_l(d_fl, f_lloc, i, j):
    f_lloc[0] = d_fl[i,j]
    
def getporos(d_poros, porosloc, i, j):
    porosloc[0] = d_poros[i,j]
    
def getKs(d_Ks, Ks, i, j):
    Ks[0] = d_Ks[i,j]

def f2m(mloc, floc):
    mloc[0] = floc[0] + floc[1] + floc[2] + floc[3] + floc[4] + floc[5] + floc[6] + floc[7] + floc[8]
    mloc[1] = -4.0*floc[0] - floc[1] - floc[2] - floc[3] - floc[4] + 2.0*floc[5] + 2.0*floc[6] + 2.0*floc[7] + 2.0*floc[8] 
    mloc[2] = 4.0*floc[0] - 2.0*floc[1] - 2.0*floc[2] - 2.0*floc[3] - 2.0*floc[4] + floc[5] + floc[6] + floc[7] + floc[8]
    mloc[3] = floc[1] - floc[3] + floc[5] - floc[6] - floc[7] + floc[8]
    mloc[4] = -2.0*floc[1] + 2.0*floc[3] + floc[5] - floc[6] - floc[7] + floc[8]
    mloc[5] = floc[2] - floc[4] + floc[5] + floc[6] - floc[7] - floc[8]
    mloc[6] = -2.0*floc[2] + 2.0*floc[4] + floc[5] + floc[6] - floc[7] - floc[8]
    mloc[7] = floc[1] - floc[2] + floc[3] - floc[4] 
    mloc[8] = floc[5] - floc[6] + floc[7] - floc[8]
    
def calc_fl_PCMloc(f_lloc, fl_PCMloc, porosloc):
    if f_lloc[0] >= 0.5:                                                        #########0.5
        fl_PCMloc[0] = porosloc[0]*f_lloc[0]   
    else:
        fl_PCMloc[0] = 0.0
        
def calc_noruloc(nor_uloc, uxloc, uyloc):   
    nor_uloc[0] = math.sqrt(uxloc[0]**2 + uyloc[0]**2) 

def calc_m_eqloc(m_eqloc, denloc, nor_uloc, fl_PCMloc, uxloc, uyloc, f_lloc):
    if f_lloc[0] >= 0.5:                                                        #########0.5
        m_eqloc[0] = denloc[0]
        m_eqloc[1] = -2.0*denloc[0] + 3.0*1.0*nor_uloc[0]**2/fl_PCMloc[0]
        m_eqloc[2] = denloc[0]-3.0*1.0*nor_uloc[0]**2.0/fl_PCMloc[0]
        m_eqloc[3] = 1.0*uxloc[0]
        m_eqloc[4] = -1.0*uxloc[0]
        m_eqloc[5] = 1.0*uyloc[0]
        m_eqloc[6] = -1.0*uyloc[0]
        m_eqloc[7] = 1.0*(uxloc[0]**2-uyloc[0]**2)/fl_PCMloc[0]
        m_eqloc[8] = 1.0*uxloc[0]*uyloc[0]/fl_PCMloc[0]    
    else:
        m_eqloc[0] = denloc[0]
        m_eqloc[1] = -2.0*denloc[0]
        m_eqloc[2] = denloc[0]
        m_eqloc[3] = 1.0*uxloc[0]
        m_eqloc[4] = -1.0*uxloc[0]
        m_eqloc[5] = 1.0*uyloc[0]
        m_eqloc[6] = -1.0*uyloc[0]
        m_eqloc[7] = 0.0
        m_eqloc[8] = 0.0

def calc_Sloc(uxloc, uyloc, Fxloc, Fyloc, fl_PCMloc, Sloc, f_lloc):
    if f_lloc[0] >= 0.5:                                                       ########### 0.5
        Sloc[0] = 0.0
        Sloc[1] = (6.0*1.0*(uxloc[0]*Fxloc[0] + uyloc[0]*Fyloc[0]))/fl_PCMloc[0]
        Sloc[2] = -(6.0*1.0*(uxloc[0]*Fxloc[0] + uyloc[0]*Fyloc[0]))/fl_PCMloc[0]
        Sloc[3] = 1.0*Fxloc[0]
        Sloc[4] = -1.0*Fxloc[0]
        Sloc[5] = 1.0*Fyloc[0]
        Sloc[6] = -1.0*Fyloc[0]
        Sloc[7] = (2.0*1.0*(uxloc[0]*Fxloc[0]-uyloc[0]*Fyloc[0]))/fl_PCMloc[0]
        Sloc[8] = (1.0*(uxloc[0]*Fyloc[0] + uyloc[0]*Fxloc[0]))/fl_PCMloc[0]           
    else:        
        Sloc[0] = 0.0
        Sloc[1] = 0.0
        Sloc[2] = 0.0
        Sloc[3] = 1.0*Fxloc[0]
        Sloc[4] = -1.0*Fxloc[0]
        Sloc[5] = 1.0*Fyloc[0]
        Sloc[6] = -1.0*Fyloc[0]
        Sloc[7] = 0.0
        Sloc[8] = 0.0

def operloc(res1loc, fuenloc, mloc, m_eqloc, Sloc, f_lloc):
#    H=256
#    tau_v = 0.5 + (0.1*H*math.sqrt(3.0*0.0208))/(math.sqrt(8.409e5))    
    if f_lloc[0] >= 0.5:
        tau_v = 0.5054481632734
    else:
        tau_v = 0.5

    res1loc[0] = 1.0*(mloc[0]-m_eqloc[0])
    res1loc[1] = 1.1*(mloc[1]-m_eqloc[1])
    res1loc[2] = 1.1*(mloc[2]-m_eqloc[2])
    res1loc[3] = 1.0*(mloc[3]-m_eqloc[3])
    res1loc[4] = 1.2*(mloc[4]-m_eqloc[4])
    res1loc[5] = 1.0*(mloc[5]-m_eqloc[5])
    res1loc[6] = 1.2*(mloc[6]-m_eqloc[6])
    res1loc[7] = (1.0/tau_v)*(mloc[7]-m_eqloc[7])
    res1loc[8] = (1.0/tau_v)*(mloc[8]-m_eqloc[8])

    fuenloc[0] = (1.0 - 0.5*1.0)*(Sloc[0])
    fuenloc[1] = (1.0 - 0.5*1.1 )*(Sloc[1])
    fuenloc[2] = (1.0 - 0.5*1.1 )*(Sloc[2])
    fuenloc[3] = (1.0 - 0.5*1.0 )*(Sloc[3])
    fuenloc[4] = (1.0 - 0.5*1.2 )*(Sloc[4])
    fuenloc[5] = (1.0 - 0.5*1.0 )*(Sloc[5])
    fuenloc[6] = (1.0 - 0.5*1.2 )*(Sloc[6])
    fuenloc[7] = (1.0 - 0.5*1.0/tau_v)*(Sloc[7])
    fuenloc[8] = (1.0 - 0.5*1.0/tau_v)*(Sloc[8])

def colision(mloc, res1loc, fuenloc):
    mloc[0] = mloc[0]-res1loc[0]+fuenloc[0]
    mloc[1] = mloc[1]-res1loc[1]+fuenloc[1]
    mloc[2] = mloc[2]-res1loc[2]+fuenloc[2]
    mloc[3] = mloc[3]-res1loc[3]+fuenloc[3]
    mloc[4] = mloc[4]-res1loc[4]+fuenloc[4]
    mloc[5] = mloc[5]-res1loc[5]+fuenloc[5]
    mloc[6] = mloc[6]-res1loc[6]+fuenloc[6]
    mloc[7] = mloc[7]-res1loc[7]+fuenloc[7]
    mloc[8] = mloc[8]-res1loc[8]+fuenloc[8]

def m2f(floc, mloc):
    floc[0]=1.11111111e-01*mloc[0]-1.11111111e-01*mloc[1]+1.11111111e-01*mloc[2]+6.93889390e-18*mloc[3]-6.93889390e-18*mloc[4]+6.93889390e-18*mloc[5]-6.93889390e-18*mloc[6]
    floc[1]=1.11111111e-01*mloc[0]-2.77777778e-02*mloc[1]-5.55555556e-02*mloc[2]+1.66666667e-01*mloc[3]-1.66666667e-01*mloc[4]+5.55111512e-17*mloc[5]+2.5e-01*mloc[7]  
    floc[2]=1.11111111e-01*mloc[0]-2.77777778e-02*mloc[1]-5.55555556e-02*mloc[2]-1.11022302e-16*mloc[4]+1.66666667e-01*mloc[5]-1.66666667e-01*mloc[6]-2.5e-01*mloc[7]
    floc[3]=1.11111111e-01*mloc[0]-2.77777778e-02*mloc[1]-5.55555556e-02*mloc[2]-1.66666667e-01*mloc[3]+1.66666667e-01*mloc[4]+1.38777878e-17*mloc[5]-2.77555756e-17*mloc[6]+2.5e-01*mloc[7]
    floc[4]=1.11111111e-01*mloc[0]-2.77777778e-02*mloc[1]-5.55555556e-02*mloc[2]-1.66666667e-01*mloc[5]+1.66666667e-01*mloc[6]-2.5e-01*mloc[7]
    floc[5]=1.11111111e-01*mloc[0]+5.55555556e-02*mloc[1]+2.77777778e-02*mloc[2]+1.66666667e-01*mloc[3]+8.33333333e-02*mloc[4]+1.66666667e-01*mloc[5]+8.33333333e-02*mloc[6]+2.5e-01*mloc[8]
    floc[6]=1.11111111e-01*mloc[0]+5.55555556e-02*mloc[1]+2.77777778e-02*mloc[2]-1.66666667e-01*mloc[3]-8.33333333e-02*mloc[4]+1.66666667e-01*mloc[5]+8.33333333e-02*mloc[6]-2.5e-01*mloc[8]
    floc[7]=1.11111111e-01*mloc[0]+5.55555556e-02*mloc[1]+2.77777778e-02*mloc[2]-1.66666667e-01*mloc[3]-8.33333333e-02*mloc[4]-1.66666667e-01*mloc[5]-8.33333333e-02*mloc[6]+2.5e-01*mloc[8]
    floc[8]=1.11111111e-01*mloc[0]+5.55555556e-02*mloc[1]+2.77777778e-02*mloc[2]+1.66666667e-01*mloc[3]+8.33333333e-02*mloc[4]-1.66666667e-01*mloc[5]-8.33333333e-02*mloc[6]-2.5e-01*mloc[8]

def setf(d_f,floc,i,j):
    d_f[i,j,0] = floc[0]
    d_f[i,j,1] = floc[1]
    d_f[i,j,2] = floc[2]
    d_f[i,j,3] = floc[3]
    d_f[i,j,4] = floc[4]
    d_f[i,j,5] = floc[5]
    d_f[i,j,6] = floc[6]
    d_f[i,j,7] = floc[7]
    d_f[i,j,8] = floc[8]
    
###############################################################################
### calulate macroscopic variables: velocity, density, strength  #############
###############################################################################
def calc_denloc(denloc, floc):
    denloc[0]=floc[0]+floc[1]+floc[2]+floc[3]+floc[4]+floc[5]+floc[6]+floc[7]+floc[8]
    
def calc_Hlsloc(H_lloc, H_sloc):
    Cpl=Cps=1.0
    T_m = 29.78
    La = Cpl*(45.0 - 20.0)/0.1241 # La = Cpl*(T_h-T_c)/St
    H_lloc[0] = Cpl*(T_m+0.5) + 1.0*La
    H_sloc[0] = Cps*(T_m-0.5) 
    
def calc_cfloc(cfloc, fl_PCMloc, f_lloc):
    if (f_lloc[0] >= 0.5):
        cfloc[0] = (1.75/(math.sqrt(175.0*fl_PCMloc[0]**3.0)))  # 2) Coeficiente inercial
    else: 
        cfloc[0] = 0.0

def getT(d_T, Tloc, i, j):
    Tloc[0]=d_T[i,j]

def calc_sigmaloc(Tloc, sigmaloc):
    T_m = 29.78  
    if Tloc[0] < T_m:
        sigmaloc[0] = 0.8352
    else:
        sigmaloc[0] = 0.8604

def calc_tau_alpha_vl(sigmaloc, tau_tloc, alf_eloc, alf_lloc, vlloc, f_lloc):
#    tau_v = 0.5 + (0.1*H*math.sqrt(3.0*0.0208))/(math.sqrt(8.409e5)) 
    if f_lloc[0] >= 0.5:    
        tau_v = 0.50544816327342                                               #HHHHHHHHHHHHHHHHHHHHHHH
    else:
        tau_v = 0.5   
        
    tau_tloc[0] = 0.5 + (0.2719*(1.0/math.sqrt(3.0))**2*(tau_v-0.5))/(1.0*sigmaloc[0]*(math.sqrt(1.0/5.0))**2.0 *0.0208) # tiempo de relajación tau_T 
    alf_eloc[0] = sigmaloc[0]*(1.0/5.0)*(tau_tloc[0]-0.5)*1.0 # difusividad efectiva 
    alf_lloc[0] = alf_eloc[0]/0.2719  # difusividad termica del líquido
    vlloc[0] = 0.0208*alf_lloc[0]   #viscosidad del líquido
     
def calc_lloc(fl_PCMloc, vlloc, cfloc, l_0loc, l_1loc, Ks):
#    H = 256.0               #longitud característica                           #HHHHHHHHHHHHHHHHHHHHHHHHH  
#    K =  1.37e-5*H**2 
    l_0loc[0] = 0.5*(1.0 + fl_PCMloc[0]*(1.0/2.0)*(vlloc[0]/Ks[0]))
    l_1loc[0] = fl_PCMloc[0]*(1.0/2.0)*(cfloc[0]/math.sqrt(Ks[0])) 
    
def calc_Gloc(vlloc, alf_lloc, Tloc, Gloc, f_lloc):
    H = 256.0                                                                  #HHHHHHHHHHHHHHHHHHHHHHHHHH
    T_0 = 20.0  
    Ra = 8.409e5 
    if f_lloc[0] >= 0.5:
        Gloc[0] = 0.0
        Gloc[1] = 0.0
        Gloc[2] = ((Ra*vlloc[0]*alf_lloc[0]*(Tloc[0] - T_0))/((45.0 - 20)*H**3))*1.0
        Gloc[3] = 0.0
        Gloc[4] = ((Ra*vlloc[0]*alf_lloc[0]*(Tloc[0] - T_0))/((45.0 - 20)*H**3))*-1.0
        Gloc[5] = ((Ra*vlloc[0]*alf_lloc[0]*(Tloc[0] - T_0))/((45.0 - 20)*H**3))*1.0
        Gloc[6] = ((Ra*vlloc[0]*alf_lloc[0]*(Tloc[0] - T_0))/((45.0 - 20)*H**3))*1.0
        Gloc[7] = ((Ra*vlloc[0]*alf_lloc[0]*(Tloc[0] - T_0))/((45.0 - 20)*H**3))*-1.0
        Gloc[8] = ((Ra*vlloc[0]*alf_lloc[0]*(Tloc[0] - T_0))/((45.0 - 20)*H**3))*-1.0
    else:
        Gloc[0]=0.0
        Gloc[1]=0.0
        Gloc[2]=0.0
        Gloc[3]=0.0
        Gloc[4]=0.0
        Gloc[5]=0.0
        Gloc[6]=0.0
        Gloc[7]=0.0
        Gloc[8]=0.0

def calc_Vloc(vxloc, vyloc, fl_PCMloc, nor_vloc, Gloc, floc):
    vxloc[0]=(floc[1]-floc[3]+floc[5]-floc[6]-floc[7]+floc[8])/1.0 + 0.5*fl_PCMloc[0]*(Gloc[5]-Gloc[6]+Gloc[7]-Gloc[8])
    vyloc[0]=(floc[2]-floc[4]+floc[5]+floc[6]-floc[7]-floc[8])/1.0 + 0.5*fl_PCMloc[0]*(Gloc[2]+Gloc[4]+Gloc[5]+Gloc[6]+Gloc[7]+Gloc[8]) # Estos valores de la velocidad ya contienen los signos de G
    nor_vloc[0] = math.sqrt(vxloc[0]**2 + vyloc[0]**2) 
    
def calc_Uloc(l_0loc, l_1loc, vxloc, vyloc, nor_vloc, uxloc, uyloc, nor_uloc, fl_PCMloc, i, j):
    uxloc[0]=vxloc[0]/(l_0loc[0] + math.sqrt(l_0loc[0]**2 + l_1loc[0]*nor_vloc[0]))
    uyloc[0]=vyloc[0]/(l_0loc[0] + math.sqrt(l_0loc[0]**2 + l_1loc[0]*nor_vloc[0]))

    if fl_PCMloc[0]==0:
        uxloc[0]=0.0
        uyloc[0]=0.0
    if i ==0 or j==0 or i==255 or j==255:                                      #HHHHHHHHHHHHHHHHHHHHHH
        uxloc[0]=0.0
        uyloc[0]=0.0

    nor_uloc[0] = math.sqrt(uxloc[0]**2 + uyloc[0]**2)

def calc_Floc(fl_PCMloc, uxloc, uyloc, cfloc, vlloc, Gloc, TFloc, Fxloc, Fyloc, Ks, i, j):
    H = 256                                                                    #HHHHHHHHHHHHHHHHHHHHHHH
#    K =  1.37e-5*H**2 
    TFloc[0] = 0.0
    TFloc[1]=-(1.0/9.0)*((fl_PCMloc[0]*vlloc[0]/Ks[0])*(uxloc[0]*1.0 + uyloc[0]*0.0)+\
             (fl_PCMloc[0]*cfloc[0]/math.sqrt(Ks[0]))*(abs(uxloc[0])*uxloc[0]*1.0+abs(uyloc[0])*uyloc[0]*0.0)) + fl_PCMloc[0]*Gloc[1]
    TFloc[2]=-(1.0/9.0)*((fl_PCMloc[0]*vlloc[0]/Ks[0])*(uxloc[0]*0.0 + uyloc[0]*1.0)+\
             (fl_PCMloc[0]*cfloc[0]/math.sqrt(Ks[0]))*(abs(uxloc[0])*uxloc[0]*0.0+abs(uyloc[0])*uyloc[0]*1.0)) + fl_PCMloc[0]*Gloc[2]
    TFloc[3]=-(1.0/9.0)*((fl_PCMloc[0]*vlloc[0]/Ks[0])*(uxloc[0]*-1.0 + uyloc[0]*0.0)+\
             (fl_PCMloc[0]*cfloc[0]/math.sqrt(Ks[0]))*(abs(uxloc[0])*uxloc[0]*-1.0+abs(uyloc[0])*uyloc[0]*0.0)) + fl_PCMloc[0]*Gloc[3]
    TFloc[4]=-(1.0/9.0)*((fl_PCMloc[0]*vlloc[0]/Ks[0])*(uxloc[0]*0.0 + uyloc[0]*-1.0)+\
             (fl_PCMloc[0]*cfloc[0]/math.sqrt(Ks[0]))*(abs(uxloc[0])*uxloc[0]*0.0+abs(uyloc[0])*uyloc[0]*-1.0)) + fl_PCMloc[0]*Gloc[4]
    TFloc[5]=-(1.0/36.0)*((fl_PCMloc[0]*vlloc[0]/Ks[0])*(uxloc[0]*1.0 + uyloc[0]*1.0)+\
             (fl_PCMloc[0]*cfloc[0]/math.sqrt(Ks[0]))*(abs(uxloc[0])*uxloc[0]*1.0+abs(uyloc[0])*uyloc[0]*1.0)) + fl_PCMloc[0]*Gloc[5]
    TFloc[6]=-(1.0/36.0)*((fl_PCMloc[0]*vlloc[0]/Ks[0])*(uxloc[0]*-1.0 + uyloc[0]*1.0)+\
             (fl_PCMloc[0]*cfloc[0]/math.sqrt(Ks[0]))*(abs(uxloc[0])*uxloc[0]*-1.0+abs(uyloc[0])*uyloc[0]*1.0)) + fl_PCMloc[0]*Gloc[6]
    TFloc[7]=-(1.0/36.0)*((fl_PCMloc[0]*vlloc[0]/Ks[0])*(uxloc[0]*-1.0 + uyloc[0]*-1.0)+\
             (fl_PCMloc[0]*cfloc[0]/math.sqrt(Ks[0]))*(abs(uxloc[0])*uxloc[0]*-1.0+abs(uyloc[0])*uyloc[0]*-1.0)) + fl_PCMloc[0]*Gloc[7]
    TFloc[8]=-(1.0/36.0)*((fl_PCMloc[0]*vlloc[0]/Ks[0])*(uxloc[0]*1.0 + uyloc[0]*-1.0)+\
             (fl_PCMloc[0]*cfloc[0]/math.sqrt(Ks[0]))*(abs(uxloc[0])*uxloc[0]*1.0+abs(uyloc[0])*uyloc[0]*-1.0)) + fl_PCMloc[0]*Gloc[8]

    
    Fxloc[0]= TFloc[1]-TFloc[3]+TFloc[5]-TFloc[6]-TFloc[7]+TFloc[8]
    Fyloc[0]= TFloc[2]-TFloc[4]+TFloc[5]+TFloc[6]-TFloc[7]-TFloc[8]
    if i==0 or j==0 or i==H or j==H:
        Fxloc[0] = 0.0
        Fyloc[0] = 0.0
 
def setvar2D(d_den, denloc, i, j):
    d_den[i,j]=denloc[0]

###############################################################################
############################## Collition step ################################
###############################################################################
def getf_2l(d_f2l, f_2lloc, i, j):
    f_2lloc[0] = d_f2l[i,j]

def getg(d_g, gloc, i, j):
    gloc[0] = d_g[i, j, 0]
    gloc[1] = d_g[i, j, 1]
    gloc[2] = d_g[i, j, 2]
    gloc[3] = d_g[i, j, 3]
    gloc[4] = d_g[i, j, 4]

def g2n(nloc, gloc):
    nloc[0] = gloc[0] + gloc[1] + gloc[2] + gloc[3] + gloc[4]    
    nloc[1] = gloc[1] - gloc[3]    
    nloc[2] = gloc[2] - gloc[4]
    nloc[3] = -4.0*gloc[0] + gloc[1] + gloc[2] + gloc[3] + gloc[4] 
    nloc[4] = gloc[1] - gloc[2] + gloc[3] - gloc[4]
    
def calc_neqloc(Tloc, uxloc, uyloc, neqloc):
    w_test = -2.0
    if Tloc[0] >= 29.78:                                                       #T_m = 29.78 por ser galio
        neqloc[0] = Tloc[0]
        neqloc[1] = uxloc[0]*Tloc[0]/0.8604                                    #sigma = 0.8604 in the liquid
        neqloc[2] = uyloc[0]*Tloc[0]/0.8604
        neqloc[3] = w_test*Tloc[0]  
        neqloc[4] = 0.0   
    else:
        neqloc[0] = Tloc[0]
        neqloc[1] = uxloc[0]*Tloc[0]/0.8352                                    #sigma = 0.8352 in the solid
        neqloc[2] = uyloc[0]*Tloc[0]/0.8352
        neqloc[3] = w_test*Tloc[0]  
        neqloc[4] = 0.0
    

def calc_tautloc(tautloc, Tloc, f_lloc):
    tau_v = 0.50544816327342                                                
#    tau_v = 0.5

    Lambda = 0.2719
    c_s = 1.0/math.sqrt(3.0)
    c_st = math.sqrt(1.0/5.0)
    Pr = 0.0208 	#Número de Prandtl
    
    if Tloc[0] < 29.78:
        tautloc[0] = 0.5 + (Lambda*c_s**2*(tau_v-0.5))/(1.0*0.8352*c_st**2.0 * Pr)
    else:
        tautloc[0] = 0.5 + (Lambda*c_s**2*(tau_v-0.5))/(1.0*0.8604*c_st**2.0 * Pr)
        
def calc_relaxloc(relaxloc, tautloc):
#    cuda.syncthreads()
    relaxloc[0] = 1.0
    relaxloc[1] = 1.0/tautloc[0]
    relaxloc[2] = 1.0/tautloc[0]    
    relaxloc[3] = 1.5
    relaxloc[4] = 1.5
    
def calc_Ssurceloc(Ssurceloc, f_lloc, f_2lloc, Tloc, porosloc):
    
    Cpl = 1.0
    La = Cpl*(45.0 - 20.0)/0.1241   # La = Cpl*(T_h-T_c)/St
    w_test = -2.00
    
    if Tloc[0] > 29.78:
        Ssurceloc[0] = -((porosloc[0]*La)/(0.8604*Cpl))*(f_lloc[0] - f_2lloc[0])/1.0
        Ssurceloc[1] = 0.0
        Ssurceloc[2] = 0.0
        Ssurceloc[3] = -w_test*((porosloc[0]*La)/(0.8604*Cpl))*(f_lloc[0] - f_2lloc[0])/1.0 
        Ssurceloc[4] = 0.0
    else:
        Ssurceloc[0] = -((porosloc[0]*La)/(0.8352*Cpl))*(f_lloc[0] - f_2lloc[0])/1.0
        Ssurceloc[1] = 0.0
        Ssurceloc[2] = 0.0
        Ssurceloc[3] = -w_test*((porosloc[0]*La)/(0.8352*Cpl))*(f_lloc[0] - f_2lloc[0])/1.0 
        Ssurceloc[4] = 0.0
       
def colis_g(nloc, neqloc, Ssurceloc, relaxloc):
    nloc[0] = nloc[0] - relaxloc[0]*(nloc[0] - neqloc[0]) + Ssurceloc[0]
    nloc[1] = nloc[1] - relaxloc[1]*(nloc[1] - neqloc[1]) + Ssurceloc[1]
    nloc[2] = nloc[2] - relaxloc[2]*(nloc[2] - neqloc[2]) + Ssurceloc[2]
    nloc[3] = nloc[3] - relaxloc[3]*(nloc[3] - neqloc[3]) + Ssurceloc[3]
    nloc[4] = nloc[4] - relaxloc[4]*(nloc[4] - neqloc[4]) + Ssurceloc[4]  

def n2g(gloc, nloc):
    gloc[0] = 0.2*nloc[0] - 0.2*nloc[3]
    gloc[1] = 0.2*nloc[0] + 0.5*nloc[1] + 0.05*nloc[3] + 0.25*nloc[4]
    gloc[2] = 0.2*nloc[0] + 0.5*nloc[2] + 0.05*nloc[3] - 0.25*nloc[4]    
    gloc[3] = 0.2*nloc[0] - 0.5*nloc[1] + 0.05*nloc[3] + 0.25*nloc[4]
    gloc[4] = 0.2*nloc[0] - 0.5*nloc[2] + 0.05*nloc[3] - 0.25*nloc[4]
    
def setg(d_g, gloc, i, j):
    d_g[i, j, 0] = gloc[0]
    d_g[i, j, 1] = gloc[1]
    d_g[i, j, 2] = gloc[2]
    d_g[i, j, 3] = gloc[3]
    d_g[i, j, 4] = gloc[4]
    
def calc_Tloc(gloc, Tloc):
    Tloc[0] = gloc[0] + gloc[1] + gloc[2] + gloc[3] + gloc[4]
    
def calc_cpfl(f_lloc, f_2lloc):
    f_2lloc[0] = f_lloc[0]

def calc_Hk(Tloc, Hkloc, f_lloc):
    Cps = 1.0   
    La = Cps*(45.0 - 20.0)/0.1241 # La = Cpl*(T_h-T_c)/St
    Hkloc[0] = Cps*Tloc[0] + f_lloc[0]*La 
    
def calc_fl(f_lloc, Hkloc):
    T_m = 29.78
    La = 1.0*(45.0 - 20.0)/0.1241 # La = Cpl*(T_h-T_c)/St
    Hl = 1.0*(T_m+0.5) + 1.0*La # liquid enthalpy
    Hs = 1.0*(T_m-0.5) # solid enthalpy  Ts=-0.02        
    if (Hkloc[0] <= Hs):
        f_lloc[0] = 0.0
    elif ((Hkloc[0] > Hs) and (Hkloc[0] < Hl)):
        f_lloc[0] = (Hkloc[0] - Hs)/(Hl - Hs)
    else:
        f_lloc[0] = 1.0