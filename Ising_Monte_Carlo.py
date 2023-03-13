import random
import time
import numpy as np
from matplotlib import pyplot as plt
from sympy import symbols, coth
from numba import jit
import copy
import math

@jit(nopython=True)
def Spin_sum(x,i,j,k,n):
    return x[(i+1)%n][j][k] + x[(i-1)%n][j][k] + x[i][(j+1)%n][k] + x[i][(j-1)%n][k] + x[i][j][(k+1)%n] + x[i][j][(k-1)%n]

@jit(nopython=True)
def Energy_and_Magnetization(x,n):
    S = 0
    M = 0
    for i in range(len(x)):
        for j in range(len(x[i])):
            for k in range(len(x[i][j])):
                S += -x[i][j][k] * (J*Spin_sum(x,i,j,k,n)) / 2   # +H
                M += x[i][j][k]
    return S,M

@jit(nopython=True)
def Monte_Carlo_step(x, T, E0, M0, n):

    i = random.randint(0,n-1)
    j = random.randint(0,n-1)
    k = random.randint(0,n-1)
    x0 = x.copy() # Записываем текущую конфигурацию

    x[i][j][k] *= -1

    #E1, M1 = Energy_and_Magnetization(x,n) # Считаем энергию для новой конфигурации

    delta_E = -2 * (J * Spin_sum(x,i,j,k,n)) * x[i][j][k]

    if delta_E > 0:
        W = np.exp(-delta_E/T)
        ksi = random.uniform(0, 1)
        if ksi <= W:
                x_new, E1, M1 = x.copy(), E0 + delta_E , M0 + 2 * x[i][j][k]
        else:
                x_new, E1, M1 = x0, E0, M0
    else:
        x_new, E1, M1 = x.copy(), E0 + delta_E, M0 + 2 * x[i][j][k]


    return x_new, E1, M1, (E1)**2, (M1)**2

@jit(nopython=True)
def Monte_Carlo_simulation_before_equil(x, T, N_eq, E0, M0, n):
        for h in range(N_eq):
                x, E0, M0, E12, M12 = Monte_Carlo_step(x, T, E0, M0,n)
        return x, E0, M0


@jit(nopython=True)
def Monte_Carlo_simulation(series, x, T, N_mc, E0, M0, n):
        E_mid = 0
        M_mid = 0
        Xi_mid = 0
        C_mid = 0
        for g in range(series):
            E = 0
            M = 0
            E2 = 0
            M2 = 0
            for h in range(N_mc):
                x, E0, M0, E12, M12 = Monte_Carlo_step(x, T, E0, M0,n)
                E += E0
                M += M0
                E2 += E12
                M2 += M12
            E_mid += E/N_mc
            M_mid += abs(M/N_mc)
            C_mid += (E2/N_mc - (E/N_mc)**2)/(T**2)
            Xi_mid += (M2/N_mc - (M/N_mc) ** 2)/T

        return  E_mid/series, M_mid/series,  C_mid/series, Xi_mid/series


######## Main
seconds = time.time()
T_list = np.arange(2, 5, 0.2) # значения температур

#k = 1.38 * 10**(-23)
J = 1
H = 0
d = 3 # размерность
n_list = [4,8,16,32]  # число узлов на одной оси
series = 3 # усреднение по сериям


for n in n_list:
    Xi_list = []
    C_list = []
    E_list = []
    M_list = []
    x = np.ones(shape=[n] * d, dtype=int)
    for i in range(len(x)):
        for j in range(len(x[i])):
            for k in range(len(x[i][j])):
                x[i][j][k] = x[i][j][k] * (-1) ** random.randint(0, 2)
    N_mc = 1000 * (n ** d) # число шагов моделирования Монте-Карло
    N_eq = 1000 * (n ** d) # число шагов до равновесия
    for T in T_list:
        E0, M0 = Energy_and_Magnetization(x,n)
        x, E0, M0 = Monte_Carlo_simulation_before_equil(x, T, N_eq, E0, M0, n)
        E,M,C,Xi = Monte_Carlo_simulation(series, x, T, N_mc, E0, M0,n)
        print(E, M, C, Xi, "Температура =", T)
        E_list.append(E)
        M_list.append(M)
        C_list.append(C)
        Xi_list.append(Xi)

    # Поиск максимальных значений восприимчивости и теплоемкости
    # for i in range(len(T_list)):
    #     if Xi_list[i] == max(Xi_list):
    #         xi = i
    #     if C_list[i] == max(C_list):
    #         c = i
    #
    # print((time.time() - seconds), n)
    # print('Теплоемкость = ', round(max(C_list),2), 'Температура = ', T_list[c], 'n='+str(n))
    # print('Намагниченность = ', round(M_list[c], 2), 'Температура = ',  T_list[c], 'n=' + str(n))
    # print('Намагниченность = ', round(M_list[xi], 2), 'Температура = ', T_list[xi], 'n=' + str(n))
    # print('Восприимчивость = ',round(max(Xi_list),2), 'Температура = ', T_list[xi], 'n='+str(n))


    fig, ax1 = plt.subplots()
    #fig.suptitle('Vertically stacked subplots')
    ax1.scatter(T_list, C_list,color='Red', s=700, label='Теплоемкость')
    fig.set_figwidth(50)
    fig.set_figheight(50)
    ax1.set_title('C(T) при n=' + str(n),fontsize= 100)
    ax1.set_xlabel('Температура', size=100)
    ax1.set_ylabel('Теплоемкость', size=100)
    plt.annotate('N_mc =' + str(N_mc), xy=(0.75, 0.80), xycoords='axes fraction',fontsize=60)
    plt.annotate('n=' + str(n), xy=(0.75, 0.75), xycoords='axes fraction',fontsize=60)
    plt.annotate('Количество серий=' + str(series), xy=(0.75, 0.70), xycoords='axes fraction',fontsize=60)
    plt.tick_params(axis='both', which='major', labelsize=100)
    name = 'Теплоемкость n=' + str(n) + '.pdf'
    fig.savefig(name, dpi=500)

    # plt.annotate('Значения T:', xy=(0, 0.50), xycoords='axes fraction', fontsize=80)
    # plt.annotate('Результаты C:', xy=(0, 0.46), xycoords='axes fraction', fontsize=80)
    # for i in range(4,len(C_list)-8):
    #     plt.annotate(str(round(T_list[i],2)), xy=(-0.25 + i * 0.1, 0.50), xycoords='axes fraction', fontsize=80)
    #     plt.annotate(str(round(C_list[i],0)), xy=(-0.25 + i * 0.1, 0.46), xycoords='axes fraction', fontsize=80)
    # fig.savefig(name, dpi = 500)

    fig, (ax4) = plt.subplots()
    ax4.scatter(T_list, M_list, color='Red', s=700)
    fig.set_figwidth(50)
    fig.set_figheight(50)
    ax4.set_title('M(T) при n=' + str(n),fontsize= 100)
    ax4.set_xlabel('Температура', size=100)
    ax4.set_ylabel('Намагниченность "', size=100)
    plt.annotate('N_mc =' + str(N_mc), xy=(0.75, 0.30), xycoords='axes fraction',fontsize=60)
    plt.annotate('n=' + str(n), xy=(0.75, 0.25), xycoords='axes fraction',fontsize=60)
    plt.annotate('Количество серий=' + str(series), xy=(0.75, 0.20), xycoords='axes fraction',fontsize=60)
    plt.tick_params(axis='both', which='major', labelsize=100)
    name = 'Намагниченность n=' + str(n) + '.pdf'
    fig.savefig(name, dpi = 500)

    fig, (ax2) = plt.subplots()
    ax2.scatter(T_list, Xi_list, color='Red', s=700, label='Восприимчивость')
    fig.set_figwidth(50)
    fig.set_figheight(50)
    ax2.set_title('Xi(T) при n=' + str(n),fontsize= 100)
    ax2.set_xlabel('Температура', size=100)
    ax2.set_ylabel('Восприимчивость', size=100)
    plt.annotate('N_mc =' + str(N_mc), xy=(0.75, 0.30), xycoords='axes fraction',fontsize=60)
    plt.annotate('n=' + str(n), xy=(0.75, 0.25), xycoords='axes fraction',fontsize=60)
    plt.annotate('Количество серий=' + str(series), xy=(0.75, 0.20), xycoords='axes fraction',fontsize=60)
    plt.tick_params(axis='both', which='major', labelsize=100)
    name = 'Восприимчивость n=' + str(n) + '.pdf'
    fig.savefig(name, dpi=500)


    fig, ax3 = plt.subplots()
    ax3.scatter(T_list, E_list, color='Red', s=700, label='Восприимчивость')
    fig.set_figwidth(50)
    fig.set_figheight(50)
    ax3.set_title('E(T) при n=' + str(n),fontsize= 100)
    ax3.set_xlabel('Температура', size=100)
    ax3.set_ylabel('Энергия', size=100)
    plt.annotate('N_mc =' + str(N_mc), xy=(0.75, 0.30), xycoords='axes fraction',fontsize=60)
    plt.annotate('n=' + str(n), xy=(0.75, 0.25), xycoords='axes fraction',fontsize=60)
    plt.annotate('Количество серий=' + str(series), xy=(0.75, 0.20), xycoords='axes fraction',fontsize=60)
    plt.tick_params(axis='both', which='major', labelsize=100)
    name = 'Энергия n=' + str(n) + '.pdf'
    fig.savefig(name, dpi = 500)




