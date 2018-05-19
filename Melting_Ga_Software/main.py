#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Melting_Ga  Copyright (C) 2018  Benjamín Salomón Noyola García, Suemi Rodriguez Romo
    This program comes with ABSOLUTELY NO WARRANTY; for details type `show w'.
    This is free software, and you are welcome to redistribute it
    under certain conditions; type `show c' for details.
"""

import os

def case_1(sub_case):
	if sub_case==1:
		os.system('python  Case_1/MRT-LB_serial_Numpy.py')
		os.system('python  Case_1/analytic_vs_num_plot.py')
	elif sub_case==2:
		os.system('python Case_1/MRT-LB_serial_numba.py')
		os.system('python Case_1/analytic_vs_num_plot.py')
	elif sub_case==3:
		os.system('python Case_1/Parallel/MRT-LB_parallel_CUDA_1.py')
		os.system('python Case_1/Parallel/analytic_vs_num_plot.py')
	elif sub_case==4:
		os.system('python Case_1/improved_parallel/MRT-LB_parallel_CUDA_2.py')
		os.system('python Case_1/improved_parallel/analytic_vs_num_plot.py')

	else:
		print "incorrect choice"

def case_2(sub_case):
	if sub_case == 1:
		os.system('cd ')
		os.system('python Case_2/Poro_homogeneous/Change_phase_poro_homo.py')
		os.system('python Case_2/Poro_homogeneous/Grafico_2D_STR.py')
	elif sub_case == 2:
		os.system('cd ')
		os.system('python Case_2/Poro_from_image/Change_phase_poro_image.py')
		os.system('python Case_2/Poro_from_image/Grafico_2D_STR.py')
	else:
		print "incorrect choice"


if __name__ == "__main__":
	"""This software simulates the phase change in 2 cases, in case 1 is 
	simulated phase change of a solid bar, in case 2 is simulated the phase
	change of the gallium embedded in a porous media. Much more details
	are in our Melting_Ga documentation or visit our web page : 
	https://suemirodriguezromo.org/recursos/   (see section 1.4.1)"""


	"""¿What case would you like to run?
    Edit the following line:"""

	case = 1   # Select between case = 1 or case = 2 #######################

	if case == 1:
		""" Case 1 can run 5 different programs:
		What case would you like to run?
		
		sub-case 1: run MRT-LB_serial_Numpy.py     Performed with Numpy
		sub-case 2: run MRT-LB_serial_Numba.py     Performed with Numba
		sub-case 3: run MRT-LB_parallel_CUDA_1.py  Performed with Numba-CUDAperformed
		sub-case 4: run MRT-LB_parallel_CUDA_2.py  Performed with Numba-CUDA (improved)
		== every case is copared with analytical solution =="""
		
		sub_case = 4    # Select between the 4 sub-cases ##################
		case_1(sub_case)

	elif case == 2:
		"""Case 2 can run 2 different cases in parallel:
		What case would you like to run?
		
		sub_case 1: run Change_phase_poro_homo.py  Performed with Numba-CUDA (homogeneous porosity)
		sub_case 2: run Change_phase_poro_image.py Performed with Numba-CUDA (porosity is obtained from an image)
		"""
		
		sub_case = 1 	# Select between sub_case = 1 or sub_case = 2#####
		case_2(sub_case)

	else:
		print "Choose between 1 or 2 cases"
