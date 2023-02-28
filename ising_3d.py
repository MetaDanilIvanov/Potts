import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
import matplotlib
import time
import multiprocessing
import os
from datetime import datetime

# wrapped-out model
# no external field
# 3d
# J = 1 interaction

# LaTeX
matplotlib.rcParams['text.latex.preamble'] = \
    r'\usepackage{amsmath} \usepackage{amssymb} \usepackage{palatino} \usepackage{textcomp}'
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Tahoma']

def mcmove(config, beta):
    """Monte Carlo move using Metropolis algorithm """
    for i in range(N):
        for j in range(N):
            for k in range(N):
                a = np.random.randint(0, N)
                b = np.random.randint(0, N)
                c = np.random.randint(0, N)
                s = config[a, b, c]
                nb = config[(a + 1) % N, b, c] + config[a, (b + 1) % N, c] \
                     + config[(a - 1) % N, b, c] + config[a, (b - 1) % N, c] \
                     + config[a, b, (c - 1) % N] + config[a, b, (c + 1) % N]
                cost = 2 * s * nb  # * J
                if cost < 0:
                    s *= -1
                elif rand() < np.exp(-cost * beta):
                    s *= -1
                config[a, b, c] = s
    return config

def calcEnergy(config):
    """Energy of a given configuration"""
    energy = np.array(np.zeros(1), dtype=np.longdouble)
    for i in range(len(config)):
        for j in range(len(config)):
            for k in range(len(config)):
                nb = config[(i + 1) % N, j, k] + config[i, (j + 1) % N, k] \
                     + config[(i - 1) % N, j, k] + config[i, (j - 1) % N, k] \
                     + config[i, j, (k + 1) % N] + config[i, j, (k - 1) % N]
                energy += -1 * nb * config[i, j, k]
    return energy / 6.
def ising_3d(calc, proc):
    if proc == 0:
        start = time.time()
    par = np.zeros(4*calc).reshape((4, calc))
    for i in range(calc):
        lists = sim_tt(N, (calc * proc + i))
        par[0][i] = lists[0]
        par[1][i] = lists[1]
        par[2][i] = lists[2]
        par[3][i] = lists[3]
        if i == 0 and proc == 0 == 0:
            print(int(((((time.time() - start)))*
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


def sim_tt(N, tt):
    n1, n2 = 1.0 / (mcSteps * N * N), 1.0 / (mcSteps * mcSteps * N * N)
    iT = 1.0 / T[tt]
    E1 = np.array(np.zeros(1), dtype=np.longdouble)
    M1 = np.array(np.zeros(1), dtype=np.longdouble)
    E2 = np.array(np.zeros(1), dtype=np.longdouble)
    M2 = np.array(np.zeros(1), dtype=np.longdouble)
    config = (2 * np.random.randint(2, size=(N, N, N)) - 1)
    for i in range(eqSteps):  # equilibrate
        mcmove(config, iT)  # Monte Carlo moves
    for i in range(mcSteps):
        mcmove(config, iT)
        Ene = calcEnergy(config)  # calculate the energy
        Mag = np.sum(config, dtype=np.longdouble)  # calculate the magnetisation
        E1 += Ene
        M1 += Mag
        M2 += (n1 * Mag * Mag)
        E2 += (n1 * Ene * Ene)
    return n1 * E1, n1 * M1, (E2 - n2 * E1 * E1) * (iT * iT), (M2 - n2 * M1 * M1) * iT


def processesed(procs, calc):
    processes = []
    for proc in range(procs):
        p = multiprocessing.Process(target=ising_3d, args=(calc, proc))
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
    Start = time.time()
    processesed(n_proc, calc)
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

    plt.savefig(f'ising_3d_MC_test_N={str(N)}.png', bbox_inches='tight', dpi=500)
    print(f'total time {int(((time.time() - Start)*1000)/60)/1000} minutes')