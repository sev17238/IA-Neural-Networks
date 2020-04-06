''' 
    UNIVERSIDAD DEL VALLE DE GUATEMALA
    INTELIGENCIA ARTIFICIAL
    HOJA DE TRABAJO NO. 2
    ---------------------------------
    DIEG SEVILLA 17238
'''

import numpy as np
from functools import reduce

# visto en clase
# Implementacion de feed forward: Encuentra las matrices de activacion de cada neurona
def feed_forward(thetas, X):
    a = [np.asarray(X)]

    for i in range(len(thetas)):
        a.append(
            sigmoide(
                np.matmul(
                    np.hstack((
                        np.ones(len(X)).reshape(len(X), 1),
                        a[i]
                    )), thetas[i].T
                )
            )
        )
    return a

# visto en clase
# Funcion que sera optimizada
def cost_function(flat_thetas, shapes, X, Y):
    a = feed_forward(
        inflate_matrixes(flat_thetas, shapes),
        X
    )
    return -(Y * np.log(a[-1]) + (1 - Y) * np.log(1 - a[-1])).sum() / len(X)

# visto en clase
# Funcion que convierte una lista a una lista de matrices la cual sera parametro para feed forward
def inflate_matrixes(flat_thetas, shapes):
    layers = len(shapes) + 1
    sizes = [shape[0] * shape[1] for shape in shapes]
    steps = np.zeros(layers, dtype=int)

    for i in range(layers - 1):
        steps[i + 1] = steps[i] + sizes[i]

    return [
        flat_thetas[steps[i]: steps[i + 1]].reshape(*shapes[i])
        for i in range(layers - 1)
    ]

# Funcion que convierte una lista de matrices a una lista
flatten_list_of_arrays = lambda list_of_arrays: reduce(
    lambda acc, v: np.array([*acc.flatten(), *v.flatten()]),
    list_of_arrays
)

# Funcion sigmoide para una matriz
# a = _____1_____
#      1 + e^-x
def sigmoide(z):
    a = [(1 / (1 + np.exp(-x))) for x in z]
    return np.asarray(a).reshape(z.shape)

# Algoritmo de back propagation para encontrar el gradiente
def back_propagation(flat_thetas, shapes, X, Y):
    m, layers = len(X), len(shapes) + 1
    thetas = inflate_matrixes(flat_thetas, shapes)
    # Paso 2.2
    a = feed_forward(thetas, X)
    # Paso 2.4
    deltas = [*range(layers - 1), a[-1] - Y]

    for i in range(layers - 2, 0, -1):
        deltas[i] = (deltas[i + 1] @ np.delete(thetas[i], 0, 1)) * (a[i] * (1 - a[i]))

    # Paso 2.5 y 3
    Deltas = []
    for i in range(layers - 1):
        Deltas.append(
            (deltas[i + 1].T
            @
            np.hstack((
                np.ones(len(a[i])).reshape(len(a[i]), 1),
                a[i]
            ))) / m
        )
    Deltas = np.asarray(Deltas)

    return flatten_list_of_arrays(
        Deltas
    )
    
# Diccionario de las prendas de ropa
mnist = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}