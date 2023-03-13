import random

import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
import matplotlib
import time
import multiprocessing
import os
from datetime import datetime

# the periodic (b = 1), antiperiodic (b = −1) and open (b = 0) boundary conditions
# with impurity (imp = 0) and without (imp = 1)
# no external field
# 2d
# J = 1 interaction

# LaTeX
matplotlib.rcParams['text.latex.preamble'] = \
    r'\usepackage{amsmath} \usepackage{amssymb} \usepackage{palatino} \usepackage{textcomp}'
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Tahoma']


def mcmove(config, beta, B):
    """Monte Carlo move using Metropolis algorithm"""
    for _ in range(N ** 2):
        i, j = np.random.randint(0, N), np.random.randint(0, N)
        nb = ((B ** (((i + 1) % N) != (i + 1))) * config[(i + 1) % N, j]
              + (B ** (((j + 1) % N) != (j + 1))) * config[i, (j + 1) % N]
              + (B ** (((i - 1) % N) != (i - 1))) * config[(i - 1), j]
              + (B ** (((j - 1) % N) != (j - 1))) * config[i, (j - 1)])
        dE = 2 * config[i, j] * (H + J * nb)
        if dE < 0 or rand() < np.exp(-dE * beta):
            config[i, j] *= -1
    return config


def calcEnergy(config, B):
    """Energy of a given configuration"""
    energy = 0
    for i in range(N):
        for j in range(N):
            energy -= J * ((B ** (((i + 1) % N) != (i + 1))) * config[(i + 1) % N, j]
                           + (B ** (((j + 1) % N) != (j + 1))) * config[i, (j + 1) % N]
                           + (B ** (((i - 1) % N) != (i - 1))) * config[(i - 1) % N, j]
                           + (B ** (((j - 1) % N) != (j - 1))) * config[i, (j - 1) % N]) * config[i, j]
    return energy / 2.


def ising_2d(calc, proc, b, imp):
    """Calculate energy, magnetisation, specific heat and magnetic susceptibility"""
    flag = True
    if proc == 0:
        start = time.time()
        flag = False
    par = np.zeros(4 * calc).reshape((4, calc))
    for i in range(calc):
        par[0][i], par[1][i], par[2][i], par[3][i] = sim_tt(N, (calc * proc + i), b, imp)
        if flag == False:
            flag = True
            left = ((time.time() - start) * ((100 / (int((1 / calc) * 10000) / 100)) - 1) / 60)
            print(f"\n{left:.2f} minutes left")
            print("сurrent Time =", datetime.now().strftime("%H:%M:%S"))
    for i in range(4):
        file = open(f"{'EMCX'[i]}_{proc}.txt", "w")
        file.write(str(par[i].tolist()))
        file.close()


def sim_tt(N, tt, b, imp):
    """Make all calculations at temperature point"""
    beta = 1.0 / T[tt]
    Ene = np.array(np.zeros(mcSteps), dtype=np.longdouble)
    Mag = np.array(np.zeros(mcSteps), dtype=np.longdouble)
    # config = (((2 - (imp == 0)) * np.random.randint(3 - imp, size=(N, N))) - (imp != 0)) - (
    #         (imp == 0) * np.ones((N, N)))
    config = (((2 * np.random.randint(2, size=(N, N))) - 1))
    for i in range(eqSteps):  # equilibrate
        config = mcmove(config, beta, b)  # Monte Carlo moves
    for i in range(mcSteps):
        config = mcmove(config, beta, b)
        Ene[i] = calcEnergy(config, b)  # calculate the energy
        Mag[i] = np.sum(config, dtype=np.longdouble)  # calculate the magnetisation
    E_mean, M_mean, C, X = np.mean(Ene), np.mean(Mag), beta ** 2 * np.std(Ene) ** 2, beta * np.std(Mag) ** 2
    return E_mean/N**2, M_mean/N**2, C/N**2, X/N**2



def processesed(procs, calc, b, imp):
    """Start multiprocessing"""
    processes = []
    for proc in range(procs):
        p = multiprocessing.Process(target=ising_2d, args=(calc, proc, b, imp))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()


# 2d
n_proc = multiprocessing.cpu_count()
it = 40
calc = it // n_proc + ((it // n_proc) != (it / n_proc))
nt = int(calc * n_proc)  # number of temperature points
eqSteps = (1000)  # 2 ** 17  # number of MC sweeps for equilibration
mcSteps = (10000)  # 2 ** 17  # number of MC sweeps for calculation
N = 10  # size of lattice
T = np.linspace(2.235, 2.255, nt)  # 2.268
H = 0
J = 1
if __name__ == "__main__":
    # print('Choose boundary condition:\nPeriodic     == 1\nAntiperiodic == -1\nOpen         == 0')
    # b = int(input('input: '))
    # print('\nDo you need impurity?\nYes == 0\nNo  == 1')
    # imp = int(input('input: '))
    b = 1
    imp = 1
    Start = time.time()
    processesed(n_proc, calc, b, imp)
    E, M, C, X = [], [], [], []
    for i in range(n_proc):
        with open(f"E_{i}.txt", "r") as f:
            E.append(eval(f.readline()))
        with open(f"M_{i}.txt", "r") as f:
            M.append(eval(f.readline()))
        with open(f"C_{i}.txt", "r") as f:
            C.append(eval(f.readline()))
        with open(f"X_{i}.txt", "r") as f:
            X.append(eval(f.readline()))
        os.remove(f"E_{i}.txt")
        os.remove(f"M_{i}.txt")
        os.remove(f"C_{i}.txt")
        os.remove(f"X_{i}.txt")
    E = [a for b in E for a in b]
    M = np.array([a for b in M for a in b])
    C = [a for b in C for a in b]
    X = [a for b in X for a in b]

    f = plt.figure(figsize=(18, 10))

    sp1 = f.add_subplot(2, 2, 1)
    plt.scatter(T, E, s=3, marker='o', color='IndianRed')
    plt.xlabel("$T$", fontsize=25, fontweight="bold")
    plt.ylabel("$E$", fontsize=25, fontweight="bold")
    plt.axis('tight')

    sp2 = f.add_subplot(2, 2, 2)
    plt.scatter(T, np.abs(M), s=3, marker='o', color='RoyalBlue')
    plt.xlabel("$T$", fontsize=25, fontweight="bold")
    plt.ylabel("$M$ ", fontsize=25, fontweight="bold")
    plt.axis('tight')

    sp3 = f.add_subplot(2, 2, 3)
    plt.scatter(T, C, s=3, marker='o', color='IndianRed')
    plt.xlabel("$T$", fontsize=25, fontweight="bold")
    plt.ylabel("$C_v$", fontsize=25, fontweight="bold")
    plt.axis('tight')

    sp4 = f.add_subplot(2, 2, 4)
    plt.scatter(T, X, s=3, marker='o', color='RoyalBlue')
    plt.xlabel("$T$", fontsize=25, fontweight="bold")
    plt.ylabel("$\chi$", fontsize=25, fontweight="bold")
    plt.axis('tight')
    #_{"OPA"[b]}_{"YN"[imp]}
    plt.savefig(f'ising_2d_MC_test_N={str(N)}.png', bbox_inches='tight', dpi=300)
    # plt.show()
    for i in range(5):
        letterr = 'EMCXT'[i]
        letter = [E, M.tolist(), C, X, T.tolist()]
        file = open(f"{letterr}.txt", "w")
        file.write(str(letter[i]))
        file.close()
    print(f'total time {((time.time() - Start) / 60):.2f} minutes')
    t_c = 2.268
