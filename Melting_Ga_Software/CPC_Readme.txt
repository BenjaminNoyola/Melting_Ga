 CPC-Librarian's Note                          05/05/2018
--------------------------------------------------------------------------------
For a complete documentation about the Melting_Ga software, please visit 
https://suemirodriguezromo.org/recursos/

General requirements:

* Have a video card (Nvidia)
* Numba
* Numba-CUDA
* Numpy 
* Sympy
* Matplotlib
* Scipy
* Spyder IDE

All these libraries are obtained when installing Anaconda 
https://www.anaconda.com/download/#linux for Python 2.7 version. 

In order to execute melting_Ga codes, the following requirements must be satisfied:

* download Anaconda  5.1:  https://www.anaconda.com/download/\#windows  
* install Anaconda through Anaconda prompt
* Update Anaconda: $ conda update conda$
* Update Numba: $ conda update  numba$
* Install CUDA: $ conda install cudatoolkit$

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
