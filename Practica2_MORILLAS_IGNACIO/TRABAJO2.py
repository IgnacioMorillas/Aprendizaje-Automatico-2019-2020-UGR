# -*- coding: utf-8 -*-
"""
TRABAJO 2
Nombre Estudiante: 
"""
import numpy as np
import matplotlib.pyplot as plt


# Fijamos la semilla
np.random.seed(1)


def simula_unif(N, dim, rango):
	return np.random.uniform(rango[0],rango[1],(N,dim))

def simula_gaus(N, dim, sigma):
    media = 0    
    out = np.zeros((N,dim),np.float64)        
    for i in range(N):
        # Para cada columna dim se emplea un sigma determinado. Es decir, para 
        # la primera columna (eje X) se usará una N(0,sqrt(sigma[0])) 
        # y para la segunda (eje Y) N(0,sqrt(sigma[1]))
        out[i,:] = np.random.normal(loc=media, scale=np.sqrt(sigma), size=dim)
    
    return out


def simula_recta(intervalo):
    points = np.random.uniform(intervalo[0], intervalo[1], size=(2, 2))
    x1 = points[0,0]
    x2 = points[1,0]
    y1 = points[0,1]
    y2 = points[1,1]
    # y = a*x + b
    a = (y2-y1)/(x2-x1) # Calculo de la pendiente.
    b = y1 - a*x1       # Calculo del termino independiente.
    
    return a, b


# EJERCICIO 1.1: Dibujar una gráfica con la nube de puntos de salida correspondiente

x = simula_unif(50, 2, [-50,50])
#CODIGO DEL ESTUDIANTE

plt.title ( " Distribucion uniforme " )
plt.xlabel ( " Eje_x " )
plt.ylabel ( " Eje_y " )
plt.scatter (x [:, 0 ], x [:, 1 ])
plt.show ()


x_g = simula_gaus(50, 2, np.array([5,7]))
#CODIGO DEL ESTUDIANTE

plt.title ( " Distribucion gausiana" )
plt.xlabel ( " Eje_x " )
plt.ylabel ( " Eje_y " )
plt.scatter (x_g [:, 0 ], x_g [:, 1 ])
plt.show ()


input("\n--- Pulsar tecla para continuar ---\n")


###############################################################################
###############################################################################
###############################################################################
# EJERCICIO 1.2: Dibujar una gráfica con la nube de puntos de salida correspondiente

# La funcion np.sign(0) da 0, lo que nos puede dar problemas
def signo(x):

	if x >= 0:
		return 1
	return -1

def f(x, y, a, b):

    return signo(y - a*x - b)

#CODIGO DEL ESTUDIANTE


a,b = simula_recta([-50,50])
signo=np.vectorize(signo)
Y = f(x[:, 0 ], x[:, 1 ], a, b)
G = np.copy(Y)
colores=['red','blue']
for i in np.unique(Y):
    pos= Y == i
    x_aux = x[pos,:]
    plt.scatter(x_aux[:,0],x_aux[:,1],label='Datos '+str(i))
    

plt.title ( " EJERCICIO 1, Apartado 2a" )
plt.xlabel ( " Eje_x " )
plt.ylabel ( " Eje_y " )
k = range(-50,50)
plt.plot( k, [a*i+b for i in k],c='black') 
plt.legend()
plt.show ()

input("\n--- Pulsar tecla para continuar ---\n")

# 1.2.b. Dibujar una gráfica donde los puntos muestren el resultado de su etiqueta, junto con la recta usada para ello
# Array con 10% de indices aleatorios para introducir ruido

#CODIGO DEL ESTUDIANTE
idx = np.random.choice(range(Y.shape[0]), size=(int(Y.shape[0]*0.1)), replace=True)
Y[idx] *= -1

colores=['red','blue']
for i in np.unique(Y):
    pos= Y == i
    x_aux = x[pos,:]
    plt.scatter(x_aux[:,0],x_aux[:,1],label='Datos '+str(i))
    
plt.title ( " EJERCICIO 1, Apartado 2b" )
plt.xlabel ( " Eje_x " )
plt.ylabel ( " Eje_y " )
k = range(-50,50)
plt.plot( k, [a*i+b for i in k],c='black') 
plt.legend()
plt.show ()

input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 1.3: Supongamos ahora que las siguientes funciones definen la frontera de clasificación de los puntos de la muestra en lugar de una recta

def plot_datos_cuad(X, y, fz, title='Point cloud plot', xaxis='x axis', yaxis='y axis'):
    #Preparar datos
    min_xy = X.min(axis=0)
    max_xy = X.max(axis=0)
    border_xy = (max_xy-min_xy)*0.01
    
    #Generar grid de predicciones
    xx, yy = np.mgrid[min_xy[0]-border_xy[0]:max_xy[0]+border_xy[0]+0.001:border_xy[0], 
                      min_xy[1]-border_xy[1]:max_xy[1]+border_xy[1]+0.001:border_xy[1]]
    grid = np.c_[xx.ravel(), yy.ravel(), np.ones_like(xx).ravel()]
    pred_y = fz(grid)
    # pred_y[(pred_y>-1) & (pred_y<1)]
    pred_y = np.clip(pred_y, -1, 1).reshape(xx.shape)
    
    #Plot
    f, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(xx, yy, pred_y, 50, cmap='RdBu',vmin=-1, vmax=1)
    ax_c = f.colorbar(contour)
    ax_c.set_label('$f(x, y)$')
    ax_c.set_ticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    ax.scatter(X[:, 0], X[:, 1], c=y, s=50, linewidth=2, 
                cmap="RdYlBu", edgecolor='white')
    
    XX, YY = np.meshgrid(np.linspace(round(min(min_xy)), round(max(max_xy)),X.shape[0]),np.linspace(round(min(min_xy)), round(max(max_xy)),X.shape[0]))
    positions = np.vstack([XX.ravel(), YY.ravel()])
    ax.contour(XX,YY,fz(positions.T).reshape(X.shape[0],X.shape[0]),[0], colors='black')
    
    ax.set(
       xlim=(min_xy[0]-border_xy[0], max_xy[0]+border_xy[0]), 
       ylim=(min_xy[1]-border_xy[1], max_xy[1]+border_xy[1]),
       xlabel=xaxis, ylabel=yaxis)
    plt.title(title)
    plt.show()
    
    
#CODIGO DEL ESTUDIANTE
    
def f1(X):
	return ((X[:, 0]-10)**2) + ((X[:, 1]-20)**2)-400

def f2(X):
	return (0.5*(X[:, 0]-10)**2) + ((X[:, 1]-20)**2)-400

def f3(X):
	return (0.5*(X[:, 0]-10)**2) - ((X[:, 1]+20)**2)-400

def f4(X):
	return X[:, 1]-20*(X[:, 0]**2)-5*X[:, 0]+3


plot_datos_cuad(x, Y,f1,'Gráfica funcion1')
plot_datos_cuad(x, Y,f2,'Gráfica funcion2')	
plot_datos_cuad(x, Y,f3,'Gráfica funcion3')
plot_datos_cuad(x, Y,f4,'Gráfica funcion4')

input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################
def coef2line(w):
    if(len(w)!= 3):
        raise ValueError('Solo se aceptan rectas para el plano 2d. Formato: [<a0>, <a1>, <b>].')

    
    a = -w[0]/w[1]
    b = -w[2]/w[1]
    
    return a, b


def plot_data(X, y, w,title='Point clod plot',x_l='x axis',y_l='y axis'):
    #Preparar datos
    a, b = coef2line(w)
    min_xy = X.min(axis=0)
    max_xy = X.max(axis=0)
    border_xy = (max_xy-min_xy)*0.01
    
    #Generar grid de predicciones
    xx, yy = np.mgrid[min_xy[0]-border_xy[0]:max_xy[0]+border_xy[0]+0.001:border_xy[0], 
                      min_xy[1]-border_xy[1]:max_xy[1]+border_xy[1]+0.001:border_xy[1]]
    grid = np.c_[xx.ravel(), yy.ravel(), np.ones_like(xx).ravel()]
    pred_y = grid.dot(w)
    pred_y = np.clip(pred_y, -1, 1).reshape(xx.shape)
    
    #Plot
    f, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(xx, yy, pred_y, 50, cmap='RdBu',
                      vmin=-1, vmax=1)
    ax_c = f.colorbar(contour)
    ax_c.set_label('$w^tx$')
    ax_c.set_ticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    ax.scatter(X[:, 0], X[:, 1], c=y, s=50, linewidth=2, 
                cmap="RdYlBu", edgecolor='white', label='Datos')
    ax.plot(grid[:, 0], a*grid[:, 0]+b, 'black', linewidth=2.0, label='Solucion')
    ax.set(
       xlim=(min_xy[0]-border_xy[0], max_xy[0]+border_xy[0]), 
       ylim=(min_xy[1]-border_xy[1], max_xy[1]+border_xy[1]),
       xlabel=x_l, ylabel=y_l)
    ax.legend()
    plt.title(title)
    plt.show()
    
# EJERCICIO 2.1: ALGORITMO PERCEPTRON

def ajusta_PLA(datos, label, max_iter, vini):
	total_iters=0
	cambios=True
	ncambios=0
	w=np.copy(vini)
	for i in range (0,max_iter):
		cambios=False
		total_iters+=1
		
		for x,y in zip(datos,label):
			if np.sign(np.dot(np.transpose(w),x))!=np.sign(y):
				w=w+(y*x)
				ncambios+=1
		if (ncambios>0):
			ncambios=0
			cambios=True
		
		if not cambios:
			break
	return w,total_iters

#CODIGO DEL ESTUDIANTE
print ("\n\nDATOS NO MODIFICADOS\n\n")
x1 = np.copy(x)
x1=np.concatenate((x1,np.ones((x1.shape[0],1),dtype=np.float64)),1)

w = np.zeros(3)

w,iters=ajusta_PLA(x1,G,1000,w)
plot_data(x1,G,w,'Perceptron con datos 2a (W iniciado a 0)')
print ("Total de iteraciones para vector iniciado a 0: ",iters)
total=0
for i in range (0,10):
	w=np.random.uniform(0,1,x1.shape[1])
	w,iters=ajusta_PLA(x1,G,1000,w)
	total+=iters
plot_data(x1,G,w,'Perceptron con datos 2a (W iniciado a aleatorio)')
print('Valor medio de iteraciones necesario para converger: {}'.format(np.mean(np.asarray(total/10))))
input("\n--- Pulsar tecla para continuar ---\n")

# Ahora con los datos del ejercicio 1.2.b

#CODIGO DEL ESTUDIANTE
print ("\n\nDATOS MODIFICADOS\n\n")
x2=np.copy(x)
x2=np.concatenate((x2,np.ones((x2.shape[0],1),dtype=np.float64)),1)
w = np.zeros(3)
w,iters=ajusta_PLA(x2,Y,1000,w)
plot_data(x2,Y,w,'Perceptron con datos 2a (W iniciado a 0)')
print ("Total de iteraciones para vector iniciado a 0: ",iters)
total=0
iters=0
for i in range (0,10):
    w=np.random.uniform(0,1,x2.shape[1])
    w,iters=ajusta_PLA(x2,Y,1000,w)
    total+=iters
plot_data(x2,Y,w,'Perceptron con datos 2a (W iniciado a aleatorio)')
print ("Media de iteraciones para vector iniciado a valores aleatoros: ",total/10)
input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 3: REGRESIÓN LOGÍSTICA CON STOCHASTIC GRADIENT DESCENT

#CODIGO DEL ESTUDIANTE
def logisticRegression(X, y, iterations = 10000, lr = 0.01):
	w = np.zeros(3)
	w_ini = np.zeros(3)
	idx = np.arange(X.shape[0])
	for i in range(iterations):
		np.random.shuffle(idx)
		X = X[idx]
		y = y[idx]
		batchX = X[0:50:]
		batchY = y[0:50:]
		for j in range(batchX.shape[0]):
			w = w - lr * function_SGD(batchX[j], batchY[j], w)
		if(np.linalg.norm(w_ini - w) < 0.01):
			return w
		w_ini = np.copy(w)
	return w

def function_SGD(X,y,w):
	a = np.dot(-y, X)	
	b = sigmoid(np.dot(np.dot(-y, np.transpose(w)), X))
	return (a * b)/X.shape[0]

def plot_datos_recta(X, y, a, b, title = 'Point clod plot', xaxis = 'x axis', yaxis = 'y axis'):
	w = line2coef(a, b)
	min_xy = X.min(axis=0)
	max_xy = X.max(axis=0)
	border_xy = (max_xy-min_xy)*0.01
	xx, yy = np.mgrid[min_xy[0]-border_xy[0]:max_xy[0]+border_xy[0]+0.001:border_xy[0], min_xy[1]-border_xy[1]:max_xy[1]+border_xy[1]+0.001:border_xy[1]]
	grid = np.c_[xx.ravel(), yy.ravel(), np.ones_like(xx).ravel()]
	pred_y = grid.dot(w)
	pred_y = np.clip(pred_y, -1, 1).reshape(xx.shape)
	f, ax = plt.subplots(figsize=(8, 6))
	contour = ax.contourf(xx, yy, pred_y, 50, cmap='RdBu', vmin=-1, vmax=1)
	ax_c = f.colorbar(contour)
	ax_c.set_label('$w^tx$')
	ax_c.set_ticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
	ax.scatter(X[:, 0], X[:, 1], c = y, s = 50, linewidth = 2, cmap = "RdYlBu", edgecolor = 'white', label = 'Datos')
	ax.plot(grid[:, 0], a*grid[:, 0]+b, 'black', linewidth = 2.0, label = 'Solucion')
	ax.set(xlim = (min_xy[0]-border_xy[0], max_xy[0]+border_xy[0]), ylim = (min_xy[1]-border_xy[1], max_xy[1]+border_xy[1]), xlabel = xaxis, ylabel = yaxis)
	ax.legend()
	plt.title(title)
	plt.show()
    
#Error del SGD
def error_SGD(w, X, Y):
	error = 0.
	Y0 = np.copy(Y)
	Y0[Y == -1] = 0
	score = np.zeros(Y.shape)
	for i in range(X.shape[0]):
		value = sigmoid(np.dot(np.dot(Y0[i], np.transpose(w)), X[i]))
		score[i] = value
		if (value <= 0.5 and Y0[i] == 1) or (value > 0.5 and Y0[i] == 0):
			error += 1.
	return error/X.shape[0], score

"""Funcion para calcular la funcion sigmoide"""
def sigmoid (x):
	return 1/(1+np.exp(-x))

def line2coef(a, b):
	w = np.zeros(3, np.float64)
	w[0] = -a
	w[1] = 1.0
	w[2] = -b
	return w

a, b = simula_recta([-50,50])
X = simula_unif(100, 2, (-50, 50))
X = np.c_[X, np.ones((X.shape[0],))]
y = np.sign(X[:, 1]-a*X[:, 0]-b)

#Se lanza la Regresion Logistica
w = logisticRegression(X, y)
#Se pintan los datos
plot_datos_recta(X, y, a, b)
#Se obtiene el error
error, y_score = error_SGD(w, X, y)
print("\nError de Clasificacion: ", error)

input("\n--- Pulsar tecla para continuar ---\n")



# Usar la muestra de datos etiquetada para encontrar nuestra solución g y estimar Eout
# usando para ello un número suficientemente grande de nuevas muestras (>999).


#CODIGO DEL ESTUDIANTE
#Simulamos los parametros (test) (misma funcion de etiquetado)
X0 = simula_unif(2000, 2, (-50, 50))
X0 = np.c_[X0, np.ones((X0.shape[0],))]
y0 = np.sign(X0[:, 1]-a*X0[:, 0]-b)
#Se pintan los datos
plot_datos_recta(X0, y0, a, b)
#Se obtiene el error
error, y_score = error_SGD(w, X0, y0)
print("Error de Clasificacion: ", error)
input("\n--- Pulsar tecla para continuar ---\n")


###############################################################################
###############################################################################
###############################################################################
#BONUS: Clasificación de Dígitos

# Funcion para leer los datos
def readData(file_x, file_y, digits, labels):
	# Leemos los ficheros	
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []	
	# Solo guardamos los datos cuya clase sea la digits[0] o la digits[1]
	for i in range(0,datay.size):
		if datay[i] == digits[0] or datay[i] == digits[1]:
			if datay[i] == digits[0]:
				y.append(labels[0])
			else:
				y.append(labels[1])
			x.append(np.array([datax[i][0], datax[i][1],1]))
			
	x = np.array(x, np.float64)
	y = np.array(y, np.float64)
	
	return x, y

# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy', [4,8], [-1,1])
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy', [4,8], [-1,1])


#mostramos los datos
fig, ax = plt.subplots()
ax.plot(np.squeeze(x[np.where(y == -1),0]), np.squeeze(x[np.where(y == -1),1]), 'o', color='red', label='4')
ax.plot(np.squeeze(x[np.where(y == 1),0]), np.squeeze(x[np.where(y == 1),1]), 'o', color='blue', label='8')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TRAINING)')
ax.set_xlim((0, 1))
plt.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(np.squeeze(x_test[np.where(y_test == -1),0]), np.squeeze(x_test[np.where(y_test == -1),1]), 'o', color='red', label='4')
ax.plot(np.squeeze(x_test[np.where(y_test == 1),0]), np.squeeze(x_test[np.where(y_test == 1),1]), 'o', color='blue', label='8')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TEST)')
ax.set_xlim((0, 1))
plt.legend()
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

#LINEAR REGRESSION FOR CLASSIFICATION 
#CODIGO DEL ESTUDIANTE
def PIA(X,Y):
	#Pseudo-inverse algorithm
	#calculo la transpuesta de X
	x_t = X.transpose()

	pseudo_inverse=np.linalg.inv(x_t.dot(X))
	pseudo_inverse=pseudo_inverse.dot(x_t)
	pseudo_inverse=pseudo_inverse.dot(Y)
	
	return pseudo_inverse
#CODIGO DEL ESTUDIANTE
def Error_lineal(X,Y,W):
	sumatory=0
	
	for i in range(X.shape[0]):
		a=W.dot(X[i])
		if a*Y[i]<0:
			sumatory=sumatory+1
			
	sumatory=sumatory/X.shape[0]
	return sumatory



#POCKET ALGORITHM
  
#CODIGO DEL ESTUDIANTE
def PLA_pocket(datos,label,max_iter,wini):
	total_iters=0
	cambios=True
	ncambios=0
	w=np.copy(wini)
	best_w=np.copy(w)
	best_ein=1
	for i in range (0,max_iter):
		cambios=False
		total_iters+=1
		
		for x,y in zip(datos,label):
			if np.sign(np.dot(np.transpose(w),x))!=np.sign(y):
				w=w+(y*x)
				ncambios+=1
		if (ncambios>0):
			ncambios=0
			cambios=True
		err=Error_lineal(datos,label,w)
		if err<best_ein:
			best_ein=err
			best_w=np.copy(w)
			
		if not cambios:
			break
		
	return best_w,total_iters

def plot_data(X, y, w,title='Point clod plot',x_l='x axis',y_l='y axis'):
    #Preparar datos
    
    a, b = coef2line(w)
    min_xy = X.min(axis=0)
    max_xy = X.max(axis=0)
    border_xy = (max_xy-min_xy)*0.01
    
    #Generar grid de predicciones
    xx, yy = np.mgrid[min_xy[0]-border_xy[0]:max_xy[0]+border_xy[0]+0.001:border_xy[0], 
                      min_xy[1]-border_xy[1]:max_xy[1]+border_xy[1]+0.001:border_xy[1]]
    grid = np.c_[xx.ravel(), yy.ravel(), np.ones_like(xx).ravel()]
    pred_y = grid.dot(w)
    pred_y = np.clip(pred_y, -1, 1).reshape(xx.shape)
    
    #Plot
    f, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(xx, yy, pred_y, 50, cmap='RdBu',
                      vmin=-1, vmax=1)
    ax_c = f.colorbar(contour)
    ax_c.set_label('$w^tx$')
    ax_c.set_ticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    ax.scatter(X[:, 0], X[:, 1], c=y, s=50, linewidth=2, 
                cmap="RdYlBu", edgecolor='white', label='Datos')
    ax.plot(grid[:, 0], a*grid[:, 0]+b, 'black', linewidth=2.0, label='Solucion')
    ax.set(
       xlim=(min_xy[0]-border_xy[0], max_xy[0]+border_xy[0]), 
       ylim=(min_xy[1]-border_xy[1], max_xy[1]+border_xy[1]),
       xlabel=x_l, ylabel=y_l)
    ax.legend()
    plt.title(title)
    plt.show()

#apartadoA
#CODIGO DEL ESTUDIANTE

w_pia=PIA(x,y)
w_pla,iters=PLA_pocket(x,y,1000,np.zeros(x.shape[1]))

plot_data(x,y,w_pla,title="PLA pocket para datos del train",x_l="Intensidad promedio.",y_l="Simetria")
plot_data(x_test,y_test,w_pla,title="PLA pocket para datos del test",x_l="Intensidad promedio.",y_l="Simetria")
plot_data(x,y,w_pia,title="Pseudo-inversa para datos del train",x_l="Intensidad promedio.",y_l="Simetria")
plot_data(x_test,y_test,w_pia,title="Pseudo-inversa para datos del test",x_l="Intensidad promedio.",y_l="Simetria")

input("\n--- Pulsar tecla para continuar ---\n")

#apartado B
print ("\n\n\nALGORITMO PLA-POCKET")
print ("\n\n\nEin: ",Error_lineal(x,y,w_pla))
print ("Etest: ",Error_lineal(x_test,y_test,w_pla))
print ("\n\n\nALGORITMO PSEUDO-INVERSA")
print ("\n\n\nEin: ",Error_lineal(x,y,w_pia))
print ("Etest: ",Error_lineal(x_test,y_test,w_pia))

input("\n--- Pulsar tecla para continuar ---\n")

#apartado C

print ("APARTADO C (0.05): ")
def VC(e,ndigitos,dim,tc):
    return e+np.sqrt((8/ndigitos)*np.log((4*((2*ndigitos)**dim +1))/tc))

eoutTrain = VC(Error_lineal(x,y,w_pla),x[:, 0].size,3,0.05)
eoutTest = VC(Error_lineal(x_test,y_test,w_pla),x_test[:, 0].size,3,0.05)

print ("\n\n\nEin: ",eoutTrain)
print ("Etest: ",eoutTest)
