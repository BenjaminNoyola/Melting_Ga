# Autor: Benjamín S. Noyola García
# Tema: gráfico de solución analítica vs numérica
from numpy import *
import matplotlib.pyplot as plt
from scipy.special import erfc
from scipy.special import erf
from scipy.optimize import brentq

def Graficar(archivoTXT, t):
	#.................. Lectura de archivo txt en columna..................#
	matriz=loadtxt(archivoTXT)#Se carga archivo correcto segun Da_x(Delta_alfa)
	ydata=zeros([200])
	for i in range(200):
		ydata[i]=matriz[3][i]
	xdata = linspace(0, 199,200)

	#.........................Solución Analítica...........................#
	#...Definición de parámetros...#
	Delta_alfa=10.0 
	T_i =-1.0
	T_b = 1.0
	T_m = 0.0
	alpha_s = 0.002
	alpha_l = Delta_alfa*alpha_s
	poros = 1.0
	sigma = 1.0
	Cpl=Cps=1.0
	rho = 1.0
	k_s=alpha_s*rho*Cps
	k_l=alpha_l*rho*Cpl
	F_0=0.01
	H=200.0
	#~ t=(F_0*H**2)/alpha_s
	#~ t=1000  #editar esta línea
	print("tiempo: ",t)
	#...Definición de una función que resuelve una ecuación no lineal ec. 45...#
	def f(n):
		La=Cpl*(T_b-T_m)
		return (exp(-n**2))/(erf(n)) + (k_s/k_l)*sqrt(alpha_l/alpha_s)*((T_i-T_m)/(T_b-T_m))* \
		(exp(-n**2*(alpha_l/alpha_s))/erfc(n*sqrt(alpha_l/alpha_s))) - ((n*La*sqrt(pi))/(Cpl*(T_b-T_m)))
	
	n=brentq(f, 0.0, 5.0) # la función brentq resuelve la ec. 45
	print(n)
	xm = 2.0*n*sqrt(alpha_l*t)
	
	print("xm = ",xm,"\n")
	
	x = linspace(0, 199, 2000)
	dominio_1 = linspace(0,1,2000)
	dominio_2 = linspace(0,1,200)
	T = zeros(2000)
	tetha   = zeros(2000)
	
	for i in range (2000):
		if i <= (2000/200.)*(xm):
			T[i]=T_b-((T_b-T_m)*erf(x[i]/(2.*sqrt(alpha_l*t))))/erf(n)
			tetha[i] = (T[i]-T_b)/(T_b-T_m)
		else:
			T[i]=T_i+((T_m-T_i)*erfc(x[i]/(2.*sqrt(alpha_s*t))))/erfc(n*sqrt(alpha_l/alpha_s))
			tetha[i] = (T[i]-T_b)/(T_b-T_m)
	
	t=str(t)
	# .............Gráfico de solución analítica vs numérica................#
	analitica,=plt.plot(dominio_1, T, 'x',lw=1, label="Analítica; t = "+t)
	numerica,=plt.plot(dominio_2 , ydata,'*',lw=1, label="MLB-TRM; t = "+t)
	
	plt.title('Solución analítica Vs Numérica')
	#plt.legend(['Ajuste','Datos laboratorio'],loc=1)
	plt.grid(True)
	plt.grid(color='g', alpha=0.5, linestyle='solid', linewidth=1.0) # intentar
	plt.ylabel('T')
	plt.text(0.4, 0.0, r"$\frac{\alpha_l }{ \alpha_s} = 10.0$", fontsize=20, color="blue")
	#~ plt.text(0.4, 0.2, r" t=1000", fontsize=30, color="blue")
	plt.ylim(-1.2, 1)
	plt.xlabel('$X/N_x$')
	plt.legend()

archivoTXT=["Temp_tiempo_1000.txt","Temp_tiempo_10000.txt","Temp_tiempo_50000.txt","Temp_tiempo_100000.txt","Temp_tiempo_150000.txt","Temp_tiempo_250000.txt","Temp_tiempo_300000.txt"]
Tiempos=[1000., 10000., 50000., 100000., 150000., 250000., 300000.]
for i in range(7):
	Graficar(archivoTXT[i], Tiempos[i])
plt.savefig("Grafico.jpg")
#plt.savefig("Grafico.eps")
plt.savefig("Grafico.png")
plt.show()
