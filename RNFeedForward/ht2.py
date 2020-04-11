''' 
    UNIVERSIDAD DEL VALLE DE GUATEMALA
    INTELIGENCIA ARTIFICIAL
    HOJA DE TRABAJO NO. 2
    ---------------------------------
    DIEG SEVILLA 17238
'''
import numpy as np # para operaciones con matrices y vectores
import pandas as pd 
from numpy import genfromtxt # para obetener los datos
from matplotlib import pyplot as plt # para graficar
from matplotlib import image as mpimg # para graficar

from pandas import DataFrame 

from AlgoritmoRN import *

import time


##train_data = genfromtxt('fashion-mnist_train.csv', delimiter=',')
train_data = pd.read_csv('fashion-mnist_train.csv')
print(train_data)

##test_data = genfromtxt('fashion-mnist_test.csv', delimiter=',')
test_data = pd.read_csv('fashion-mnist_test.csv')
print(test_data)

NORMALIZADOR = 1000.0


x_tr = train_data.iloc[:, 1:] / NORMALIZADOR
m_tr, n_tr = x_tr.shape

x_t = test_data.iloc[:, 1:] / NORMALIZADOR
m_t, n_t = x_t.shape

X = np.vstack((
    x_tr,
    x_t
))
m, n = X.shape


y_tr = np.asarray(train_data.iloc[:, 0])
y_tr = y_tr.reshape(m_tr,1)

y_t = np.asarray(test_data.iloc[:, 0])
y_t = y_t.reshape(m_t,1)

## 
y = np.vstack((
    y_tr,
    y_t
))

y = y.reshape(m, 1)

# Matriz de categorizacion
Y = (y == np.array(range(10))).astype(int)

# Estructura de la red neuronal
NETWORK_ARCHITECTURE = np.array([
    n,
    130,
    10
])

# Funcion para los shapes de las thetas
theta_shapes = np.hstack((
    NETWORK_ARCHITECTURE[1:].reshape(len(NETWORK_ARCHITECTURE) - 1, 1),
    (NETWORK_ARCHITECTURE[:-1] + 1).reshape(len(NETWORK_ARCHITECTURE) - 1, 1)
))

# Convierte la matriz a un array de thetas
flat_thetas = flatten_list_of_arrays([
    np.random.rand(*theta_shape)
    for theta_shape in theta_shapes
])