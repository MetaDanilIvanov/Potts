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
# 3d
# J = 1 interaction

# LaTeX
matplotlib.rcParams['text.latex.preamble'] = \
    r'\usepackage{amsmath} \usepackage{amssymb} \usepackage{palatino} \usepackage{textcomp}'
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Tahoma']


def mcmove(config, beta, B):
    """Monte Carlo move using Metropolis algorithm """
    for a in range(N):
        for b in range(N):
            for c in range(N):
                i = np.random.randint(0, N)
                j = np.random.randint(0, N)
                k = np.random.randint(0, N)
                s = config[i, j, k]
                nb = (B**(((i + 1) % N) != (i+1)))*config[(i + 1) % N, j, k]\
                     + (B**(((j + 1) % N) != (j+1)))* config[i, (j + 1) % N, k] \
                     + (B**(((i - 1) % N) != (i-1)))* config[(i - 1), j, k]\
                     + (B**(((j- 1) % N) != (j-1)))* config[i, (j - 1), k] \
                     + (B**(((k + 1) % N) != (k+1)))* config[i, j, (k + 1) % N]\
                     + (B**(((k- 1) % N) != (k-1)))* config[i, j, (k - 1)]
                cost = 2 * s * nb  # * J
                if cost < 0:
                    s *= -1
                elif rand() < np.exp(-cost * beta):
                    s *= -1
                config[i, j, k] = s
    return config


def calcEnergy(config, B):
    """Energy of a given configuration"""
    energy = np.array(np.zeros(1), dtype=np.longdouble)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                nb = (B**(((i + 1) % N) != (i+1)))*config[(i + 1) % N, j, k]\
                     + (B**(((j + 1) % N) != (j+1)))* config[i, (j + 1) % N, k] \
                     + (B**(((i - 1) % N) != (i-1)))* config[(i - 1), j, k]\
                     + (B**(((j- 1) % N) != (j-1)))* config[i, (j - 1), k] \
                     + (B**(((k + 1) % N) != (k+1)))* config[i, j, (k + 1) % N]\
                     + (B**(((k- 1) % N) != (k-1)))* config[i, j, (k - 1)]
                energy += -1 * nb * config[i, j, k]
    return energy / 6.


def ising_3d(calc, proc, b, imp):
    if proc == 0:
        start = time.time()
    par = np.zeros(4 * calc).reshape((4, calc))
    for i in range(calc):
        lists = sim_tt(N, (calc * proc + i), b, imp)
        par[0][i] = lists[0]
        par[1][i] = lists[1]
        par[2][i] = lists[2]
        par[3][i] = lists[3]
        if i == 0 and proc == 0 == 0:
            print(int(((((time.time() - start))) *
                       ((100 / (int((1 / calc) * 10000) / 100)) - 1) / 60) * 100) / 100,
                  'minutes left')
            print("Current Time =", datetime.now().strftime("%H:%M:%S"))
    file = open(f"E_{proc}.txt", "w")
    file.write(str(par[0].tolist()))
    file.close()
    file = open(f"M_{proc}.txt", "w")
    file.write(str(par[1].tolist()))
    file.close()
    file = open(f"C_{proc}.txt", "w")
    file.write(str(par[2].tolist()))
    file.close()
    file = open(f"X_{proc}.txt", "w")
    file.write(str(par[3].tolist()))
    file.close()


def sim_tt(N, tt, b, imp):
    n1, n2 = 1.0 / (mcSteps * N * N), 1.0 / (mcSteps * mcSteps * N * N)
    iT = 1.0 / T[tt]
    E1 = np.array(np.zeros(1), dtype=np.longdouble)
    M1 = np.array(np.zeros(1), dtype=np.longdouble)
    E2 = np.array(np.zeros(1), dtype=np.longdouble)
    M2 = np.array(np.zeros(1), dtype=np.longdouble)
    config = (((2 - (imp == 0)) * np.random.randint(3 - imp, size=(N, N, N))) - (imp != 0)) - (
                (imp == 0) * np.ones((N, N, N)))
    for i in range(eqSteps):  # equilibrate
        mcmove(config, iT, b)  # Monte Carlo moves
    for i in range(mcSteps):
        mcmove(config, iT, b)
        Ene = calcEnergy(config, b)  # calculate the energy
        Mag = np.sum(config, dtype=np.longdouble)  # calculate the magnetisation
        E1 += Ene
        M1 += Mag
        M2 += (n1 * Mag * Mag)
        E2 += (n1 * Ene * Ene)
    return n1 * E1, n1 * M1, (E2 - n2 * E1 * E1) * (iT * iT), (M2 - n2 * M1 * M1) * iT


def processesed(procs, calc, b, imp):
    processes = []
    for proc in range(procs):
        p = multiprocessing.Process(target=ising_3d, args=(calc, proc, b, imp))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()


# 3d
n_proc = multiprocessing.cpu_count()
it = 300
calc = it // n_proc + ((it // n_proc) != (it / n_proc))
nt = int(calc * n_proc)  # number of temperature points
eqSteps = 2 ** 9  # number of MC sweeps for equilibration
mcSteps = 2 ** 9  # number of MC sweeps for calculation
N = 10  # size of lattice
T = np.linspace(2.5, 5.5, nt)  # 4.5
if __name__ == "__main__":
    print('Choose boundary condition:\nPeriodic     == 1\nAntiperiodic == -1\nOpen         == 0')
    b = int(input('input: '))
    print('\nDo you need impurity?\nYes == 0\nNo  == 1')
    imp = int(input('input: '))
    Start = time.time()
    processesed(n_proc, calc, b, imp)
    E = []
    M = []
    C = []
    X = []
    E_list = []
    M_list = []
    C_list = []
    X_list = []
    for i in range(n_proc):
        E_list.append(f"E_{i}.txt")
        M_list.append(f"M_{i}.txt")
        C_list.append(f"C_{i}.txt")
        X_list.append(f"X_{i}.txt")
    for i in range(n_proc):
        with open(f"E_{i}.txt", "r") as f:
            E.append(eval(f.readline()))
        with open(f"M_{i}.txt", "r") as f:
            M.append(eval(f.readline()))
        with open(f"C_{i}.txt", "r") as f:
            C.append(eval(f.readline()))
        with open(f"X_{i}.txt", "r") as f:
            X.append(eval(f.readline()))
    E = [a for b in E for a in b]
    M = np.array([a for b in M for a in b])
    C = [a for b in C for a in b]
    X = [a for b in X for a in b]
    for i in range(n_proc):
        os.remove(f"E_{i}.txt")
        os.remove(f"M_{i}.txt")
        os.remove(f"C_{i}.txt")
        os.remove(f"X_{i}.txt")

    f = plt.figure(figsize=(18, 10))

    sp1 = f.add_subplot(2, 2, 1)
    plt.scatter(T, E, s=3, marker='o', color='IndianRed')
    plt.xlabel("$T$", fontsize=25)
    plt.ylabel("$E$", fontsize=25)
    plt.axis('tight')

    sp2 = f.add_subplot(2, 2, 2)
    plt.scatter(T, abs(M), s=3, marker='o', color='RoyalBlue')
    plt.xlabel("$T$", fontsize=25)
    plt.ylabel("$M$ ", fontsize=25)
    plt.axis('tight')

    sp3 = f.add_subplot(2, 2, 3)
    plt.scatter(T, C, s=3, marker='o', color='IndianRed')
    plt.xlabel("$T$", fontsize=25)
    plt.ylabel("$C_v$", fontsize=25)
    plt.axis('tight')

    sp4 = f.add_subplot(2, 2, 4)
    plt.scatter(T, X, s=3, marker='o', color='RoyalBlue')
    plt.xlabel("$T$", fontsize=25)
    plt.ylabel("$\chi$", fontsize=25)
    plt.axis('tight')
    impurity = 'YN'
    BC = 'OPA'
    plt.savefig(f'ising_2d_MC_test_N={str(N)}_{BC[b]}_{impurity[imp]}.png', bbox_inches='tight', dpi=500)
    print(f'total time {int(((time.time() - Start)*1000)/60)/1000} minutes')
