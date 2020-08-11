Melting Ga software
--------------------------------------------------------------------------------
For a complete documentation about the Melting_Ga software. We explain 
with detail in paper: Simulations of Gallium melting based 
on multiple-relaxation time lattice Boltzmann method 
performed with CUDA in Python, Suemi Rodriguez Romo, Benjamin Noyola, 2020.

** In this study we present two cases: **


This software was tested on python 3.8, and is divided in 2 cases 
which are in 2 files. Please, go the readme.MD in every case. 
Select the case than you want to run.

* Case 1:  Phase change simulation of a simple bar of a solid material 
(caused by thermal phenomena). No porous medium is involved.

1.1 Simulation using numpy (case 1)
1.2 Simulation using CUDA (case 1)
1.3 Analityc solution to compare with CUDA or numpy solutions (case 1) 

* Case 2: Solid-liquid phase change simulation of a material (Ga) 
immersed in a porous media, performed by Numba-CUDA. This simulation 
uses MRLBM, with a D2Q5 stencil for the heat transfer in the system and 
a D2Q9 stencil for the momentum transfers in the liquid Ga. The last 
one takes into account  natural convection induced by gravity.

2.1 CUDA solution using homogeneous porosity (case 2)
2.2 CUDA solution using heterogeneous porosity (case 2)

** Install requirements **
CUDA drivers and numba is required
<Install CUDA: $ conda install cudatoolkit$>

In orther to install all require libraries, run the following command:

    <pip intall requirements.txt>


** Run Simulation **
Go to the corresponding case (1 or 2), read "readme.MD" and run the 
simulation


