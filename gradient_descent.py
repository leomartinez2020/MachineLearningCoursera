# Gradient descent implementation using pandas
# Data from Andrew Ng's course on machine learning

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

archivo = 'ex1data1.txt'

df = pd.read_csv(archivo, header=None, names=['population', 'profit'])

def costo(dframe, th0, th1):
    return np.sum(np.square((dframe['population'] * th1 + th0) - dframe['profit'])) / len(dframe) / 2.0

def grad_desc(dframe, th0, th1, alpha):
    length = len(dframe)
    dframe['pred'] = dframe['population'] * th1 + th0
    th0 = th0 - alpha / length * np.sum((dframe['pred'] - dframe['profit']))
    th1 = th1 - alpha / length * np.sum(((dframe['pred'] - dframe['profit']) * dframe['population']))
    return th0, th1


theta0, theta1 = 0, 0
alpha = 0.01

def test_graph(df, iteraciones):
    theta0, theta1 = 0, 0
    for elem in range(iteraciones):
        theta0, theta1 = grad_desc(df, theta0, theta1, alpha)

    cost = costo(df, theta0, theta1)

    plt.plot(df['population'], df['profit'], 'b.', df['population'], df['population']*theta1 + theta0, 'r-')
    plt.title('Método del gradiente con pandas')
    plt.xlabel('Población')
    plt.ylabel('Ganancia')
    plt.text(5, 23, 'Costo: {}'.format(cost))
    plt.text(5, 19, 'theta0: {}\ntheta1: {}\niteraciones: {}'.format(theta0, theta1, iteraciones))
    plt.show()

def plot_with_cost(df, iteraciones):
    cont_costo = []
    theta0, theta1 = 0, 0
    for elem in range(iteraciones):
        theta0, theta1 = grad_desc(df, theta0, theta1, alpha)
        if elem % 1 == 0:
            cost = costo(df, theta0, theta1)
            cont_costo.append(cost)
    plt.plot(range(1, iteraciones + 1, 1), cont_costo)
    plt.title('Gráfico de disminución del costo')
    plt.xlabel('Iteraciones')
    plt.ylabel('Costo J')
    plt.show()

def oneplot(df, iteraciones):
    theta0, theta1 = 0, 0
    for elem in range(iteraciones):
        theta0, theta1 = grad_desc(df, theta0, theta1, alpha)

    cost = costo(df, theta0, theta1)

    plt.plot(df['population'], df['profit'], 'b.', df['population'], df['population']*theta1 + theta0, 'r-')
    plt.text(5, 23, 'Costo: {}'.format(cost))
    plt.text(5, 19, 'iteraciones: {}'.format(iteraciones))


def multiple_plots(df):
    plt.figure(1)
    plt.subplot(221)
    oneplot(df, 0)
    plt.subplot(222)
    oneplot(df, 1)
    plt.subplot(223)
    oneplot(df, 10)
    plt.subplot(224)
    oneplot(df, 500)
    plt.show()

multiple_plots(df)
#test_graph(df, 1500)
#plot_with_cost(df, 1500)
