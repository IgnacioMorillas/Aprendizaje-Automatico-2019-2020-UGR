# -*- coding: utf-8 -*-
"""
Created on Sat May 18 20:00:39 2019

@author: nacho
"""
from __future__ import print_function
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Lasso, Ridge
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import itertools


random_seed = 20000913
seed = 20000913

"""
*************************************************************************
*************************************************************************
                    CLASIFICACIÓN
*************************************************************************
*************************************************************************
"""

###############################################################
# Funcion Encargada de la clasificación
###############################################################

def Clasificar():
    Train_X, Train_Y, Test_X, Test_Y = Cargar_optdigits()
    model = preprocesamiento_optdigits(Train_X, Train_Y, 1000)

    print("Best parameters set found on development set:")
    print()
    print(model.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = model.cv_results_['mean_test_score']
    stds = model.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, model.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        
    #print("\n\n\n END OF TUNNING PARAMETERS!!!\n\n\n")

    print("The model is trained on the full train set and with best parameters")
    best_logistic_model = LogisticRegression(**model.best_params_)
    best_logistic_model.fit(Train_X, Train_Y)
    
    print("The scores are computed with full test set")

    y_true, y_pred = Test_Y, best_logistic_model.predict(Test_X)
    print(classification_report(y_true, y_pred))
    print()
    
    matrix = confusion_matrix(Test_Y,y_pred)
    plot_confusion_matrix(matrix, np.unique(Test_Y))
    
###############################################################
#Función encargada de cargar los Datos de entrenamiento y Test
###############################################################

def Cargar_optdigits():
    train_x = np.float64(np.load("datos/optdigits_tra_X.npy"))
    train_y = np.float64(np.load("datos/optdigits_tra_y.npy"))
    test_x = np.float64(np.load("datos/optdigits_tes_X.npy"))
    test_y = np.float64(np.load("datos/optdigits_tes_y.npy"))
    
    return train_x, train_y, test_x, test_y

###############################################################
# Pinta la Matriz de Confusion
###############################################################

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	"""
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')

	print(cm)

	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.show()

###############################################################
# Funcion para el preprocesamiento de Datos
###############################################################

def preprocesamiento_optdigits(X, y, Iterations):
    MinMaxScaler().fit_transform(X)
    tuned_parameters = [{'penalty': ['l1'], 'C': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0], 'tol':[1e-3, 1e-4, 1e-5, 1e-6, 1e-7]},
					{'penalty': ['l2'], 'C': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0], 'tol':[1e-3, 1e-4, 1e-5, 1e-6, 1e-7]}]
    clf = GridSearchCV(LogisticRegression(random_state = random_seed, max_iter = Iterations), tuned_parameters, cv = 5, scoring = 'accuracy')
    return clf.fit(X, y)


"""
*************************************************************************
*************************************************************************
                    REGRESIÓN
*************************************************************************
*************************************************************************
"""

###############################################################
# Funcion Encargada de la Regresion
###############################################################

def Regresion():
    Train_X, Train_Y, Test_X, Test_Y = Cargar_airfoil()
    Train_X, Test_X=preprocesamiento_airfoil_Lasso(Train_X, Test_X)
    ridge_Lasso_model(Train_X, Train_Y,Test_X, Test_Y)
    
###############################################################
# Funcion Encargada de Cargar los datos airfoil
###############################################################

def Cargar_airfoil():
    X = np.load('datos/airfoil_self_noise_X.npy')
    y = np.load('datos/airfoil_self_noise_y.npy')
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.20, random_state = seed)
    return train_x, train_y, test_x, test_y

###############################################################
# Funcion Encargada del preprocesamiento
###############################################################

def preprocesamiento_airfoil_Lasso(X_Train, X_Test):
    
    pipe = Pipeline([('Polynomial', preprocessing.PolynomialFeatures(degree = 6)), ('Scale', preprocessing.MinMaxScaler())])
    pipe.fit(X_Train)
    X_Train = pipe.transform(X_Train)
    X_Test = pipe.transform(X_Test)
    return X_Train, X_Test
    
def ridge_Lasso_model(X_Train, Y_Train,Test_X, Test_Y):
    tuned_parameters_Ridge = [{'alpha': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0], 'tol':[1e-3, 1e-4, 1e-5, 1e-6, 1e-7]}]
    tuned_parameters_Lasso = [{'alpha': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0], 'selection':['random', 'cyclic'], 'tol':[1e-3, 1e-4, 1e-5, 1e-6, 1e-7]}]
    
    bestLassoParams = gridSearchFunction(Lasso(random_state=seed), tuned_parameters_Lasso, 'neg_mean_squared_error',X_Train, Y_Train)
    
    bestLasso = Lasso(**bestLassoParams).fit(X_Train, Y_Train)
    
    print("\n************************************************")
    print("\n Regresion con Lasso: \n")
    print("\nRidge Resultado con r2: %s" % bestLasso.score(Test_X, Test_Y))
    print("Ridge Resultado con MSE: %s" % mean_squared_error(Test_Y, bestLasso.predict(Test_X)),"\n\n")
    print("\n************************************************")
    input("\n\nPulsa Enter para continuar la ejecucion:\n")
    bestRidgeParams = gridSearchFunction(Ridge(random_state=seed), tuned_parameters_Ridge, 'neg_mean_squared_error',X_Train, Y_Train)

    bestRidge = Ridge(**bestRidgeParams).fit(X_Train, Y_Train)

    print("\n************************************************")
    print("\n Regresion con Ridge: \n")
    print("\nRidge Resultado con r2: %s" % bestRidge.score(Test_X, Test_Y))
    print("Ridge Resultado con MSE: %s" % mean_squared_error(Test_Y, bestRidge.predict(Test_X)),"\n\n")
    print("\n************************************************")

    
def gridSearchFunction(Model, parameters, scoring,X_train, y_train):
	clf = GridSearchCV(Model, parameters, cv=5, scoring=scoring)
	clf.fit(X_train, y_train)

	print("Best parameters set found on development set:")
	print()
	print(clf.best_params_)
	print()
	print("Grid scores on development set:")
	print()
	means = clf.cv_results_['mean_test_score']
	stds = clf.cv_results_['std_test_score']
	for mean, std, params in zip(means, stds, clf.cv_results_['params']):
		print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
	#print("\n\n\n END OF TUNNING PARAMETERS!!!\n\n\n")

	print("The model is trained on the full train set and with best parameters")
	return clf.best_params_

###############################################################
# Funcion Principal (Main)
###############################################################

def main():
    np.random.seed(random_seed)
    
    print("\n Clasificacion: \n")
    Clasificar()
    input("\nPulsa Enter para continuar la ejecucion:")
    
    print("\n Regresion: \n")
    Regresion()

    
if __name__ == "__main__":
    main()