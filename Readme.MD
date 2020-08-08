Melting Ga software
--------------------------------------------------------------------------------
For a complete documentation about the Melting_Ga software. we explain 
in detail in paper: Simulations of Gallium melting based 
on multiple-relaxation time lattice Boltzmann method 
performed with CUDA in Python, Suemi Rodriguez Romo, 
Benjamin Noyola, 2020.


***In this study we present two cases:***

Update Numba: $ conda update  numba$
Install CUDA: $ conda install cudatoolkit$

This software is divided in 2 cases which are in 2 files, please, read 
the readme in every case. Execute main program (main.py) and select the 
case and sub_case you want to run, (see section 1.4.1 in melting_Ga 
documentation).

* Case 1:  Phase change simulation of a simple bar of a solid material 
(caused by thermal phenomena). No porous medium is involved.

* Case 2: Solid-liquid phase change simulation of a material (Ga) 
immersed in a porous media, performed by Numba-CUDA. This simulation 
uses MRLBM, with a D2Q5 stencil for the heat transfer in the system and 
a D2Q9 stencil for the momentum transfers in the liquid Ga. The last 
one takes into account  natural convection induced by gravity.

1.1 Simulation using numpy (case 1)
1.2 Simulation using CUDA (case 1)
1.3 Analityc solution to compare with CUDA or numpy solutions (case 1)
2.1 CUDA solution using homogeneous porosity (case 2)
2.2 CUDA solution using heterogeneous porosity (case 2)

***Install requirements***

* Install CUDA drivers

In orther to install all require libraries, run the following command:

	pip intall requirements.txt


*Run Simulation*

Run the simulation with the following commands

	python <main_pakage.py>
