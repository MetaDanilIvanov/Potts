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
    for a in range(N):
        for b in range(N):
            i, j = np.random.randint(0, N), np.random.randint(0, N)
            s = config[i, j]
            cost = 2 * s * ((B ** (((i + 1) % N) != (i + 1))) * config[(i + 1) % N, j]
                            + (B ** (((j + 1) % N) != (j + 1))) * config[i, (j + 1) % N]
                            + (B ** (((i - 1) % N) != (i - 1))) * config[(i - 1), j]
                            + (B ** (((j - 1) % N) != (j - 1))) * config[i, (j - 1)])
            if cost < 0:
                s *= -1
            elif rand() < np.exp(-cost * beta):
                s *= -1
            config[i, j] = s


def calcEnergy(config, B):
    """Energy of a given configuration"""
    energy = np.array(np.zeros(1), dtype=np.longdouble)
    for i in range(N):
        for j in range(N):
            energy += -((B ** (((i + 1) % N) != (i + 1))) * config[(i + 1) % N, j]
                        + (B ** (((j + 1) % N) != (j + 1))) * config[i, (j + 1) % N]
                        + (B ** (((i - 1) % N) != (i - 1))) * config[(i - 1), j]
                        + (B ** (((j - 1) % N) != (j - 1))) * config[i, (j - 1)]) * config[i, j]
    return energy / 4.


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
            print('\nabout', int(((((time.time() - start))) *
                             ((100 / (int((1 / calc) * 10000) / 100)) - 1) / 60) * 100) / 100,
                  'minutes left')
            print("current Time =", datetime.now().strftime("%H:%M:%S"))
    for i in range(4):
        file = open(f"{'EMCX'[i]}_{proc}.txt", "w")
        file.write(str(par[i].tolist()))
        file.close()


def sim_tt(N, tt, b, imp):
    """Make all calculations at temperature point"""
    n1, n2 = 1.0 / (mcSteps * N * N), 1.0 / (mcSteps * mcSteps * N * N)
    iT = 1.0 / T[tt]
    E1 = np.array(np.zeros(1), dtype=np.longdouble)
    M1 = np.array(np.zeros(1), dtype=np.longdouble)
    E2 = np.array(np.zeros(1), dtype=np.longdouble)
    M2 = np.array(np.zeros(1), dtype=np.longdouble)
    config = (((2 - (imp == 0)) * np.random.randint(3 - imp, size=(N, N))) - (imp != 0)) - (
            (imp == 0) * np.ones((N, N)))
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
it = 3000
calc = it // n_proc + ((it // n_proc) != (it / n_proc))
nt = int(calc * n_proc)  # number of temperature points
eqSteps = 2 ** 10  # number of MC sweeps for equilibration
mcSteps = 2 ** 10  # number of MC sweeps for calculation
N = 5  # size of lattice
T = np.linspace(1.2, 3.5, nt)  # 2.25
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
    plt.savefig(f'ising_2d_MC_test_N={str(N)}_{"OPA"[b]}_{"YN"[imp]}.png', bbox_inches='tight', dpi=300)
    for i in range(4):
        letterr = 'EMCX'[i]
        letter = [E, M.tolist(), C, X]
        file = open(f"{letterr}.txt", "w")
        file.write(str(letter[i]))
        file.close()
    file = open("T.txt", "w")
    file.write(str(T.tolist()))
    file.close()
    print(f'total time {int(((time.time() - Start) * 1000) / 60) / 1000} minutes')
    t_c = T[C.index(max(C))]
    print(t_c)