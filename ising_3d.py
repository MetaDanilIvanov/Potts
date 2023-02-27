import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
import matplotlib
import time
import multiprocessing

# wrapped-out model
# no external field
# 3d
# J = 1 interaction
# LaTeX
matplotlib.rcParams['text.latex.preamble'] = \
    r'\usepackage{amsmath} \usepackage{amssymb} \usepackage{palatino} \usepackage{textcomp}'
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Tahoma']


def initial_state(N: int):
    """generates a random spin configuration for initial condition 3d"""
    state = []
    for levels in range(N):
        slicee = 2 * np.random.randint(2, size=(N, N)) - 1
        state.append(slicee.tolist())
    return np.array(state)


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
    energy = 0
    for i in range(len(config)):
        for j in range(len(config)):
            for k in range(len(config)):
                nb = config[(i + 1) % N, j, k] + config[i, (j + 1) % N, k] \
                     + config[(i - 1) % N, j, k] + config[i, (j - 1) % N, k] \
                     + config[i, j, (k + 1) % N] + config[i, j, (k - 1) % N]
                energy += -1 * nb * config[i, j, k]
    return energy / 6.


def calcMag(config):
    """Magnetization of a given configuration"""
    mag = np.sum(config)
    return mag
def sim_tt(calc, proc, nt, N):
    T = np.linspace(2., 7., nt)  # 4.5
    E, M, C, X = np.array(np.zeros(nt), dtype=np.float64), np.array(np.zeros(nt), dtype=np.float64), np.array(
        np.zeros(nt), dtype=np.float64), np.array(np.zeros(nt), dtype=np.float64)
    n1, n2 = 1.0 / (mcSteps * N * N), 1.0 / (mcSteps * mcSteps * N * N)
    for tt in range(nt):
        iT = 1.0 / T[tt]
        E1 = np.array(np.zeros(1), dtype=np.float64)
        M1 = np.array(np.zeros(1), dtype=np.float64)
        E2 = np.array(np.zeros(1), dtype=np.float64)
        M2 = np.array(np.zeros(1), dtype=np.float64)  ###
        config = initial_state(N)
        for i in range(eqSteps):  # equilibrate
            mcmove(config, iT)  # Monte Carlo moves
        for i in range(mcSteps):
            mcmove(config, iT)
            Ene = calcEnergy(config)  # calculate the energy
            Mag = calcMag(config)  # calculate the magnetisation
            E1 += Ene
            M1 += Mag
            M2 += (Mag * Mag)
            E2 += (Ene * Ene)
        E[tt] = n1 * E1
        M[tt] = n1 * M1
        C[tt] = (n1 * E2 - n2 * E1 * E1) * (iT * iT)
        X[tt] = (n1 * M2 - n2 * M1 * M1) * iT

def ising_3d(N, nt, eqSteps, mcSteps, calc, proc):
    # start_time = time.time()
    # N = NN[n - 1] # size of the lattice, N x N
    # N = 150
    # J = 5  # interaction for strange reason kills the code
    # no external field
    # ttt = time.time()
    # c = 0
    # between = 0
    T = np.linspace(2., 7., nt)
    for tt in range(nt):
        iT = 1.0 / T[tt]
        E1 = np.array(np.zeros(1), dtype=np.float64)
        M1 = np.array(np.zeros(1), dtype=np.float64)
        E2 = np.array(np.zeros(1), dtype=np.float64)
        M2 = np.array(np.zeros(1), dtype=np.float64)  ###
        config = initial_state(N)
        for i in range(eqSteps):  # equilibrate
            mcmove(config, iT)  # Monte Carlo moves
        for i in range(mcSteps):
            mcmove(config, iT)
            Ene = calcEnergy(config)  # calculate the energy
            Mag = calcMag(config)  # calculate the magnetisation
            E1 += Ene
            M1 += Mag
            M2 += (Mag * Mag)
            E2 += (Ene * Ene)
        E[tt] = n1 * E1
        M[tt] = n1 * M1
        C[tt] = (n1 * E2 - n2 * E1 * E1) * (iT * iT)
        X[tt] = (n1 * M2 - n2 * M1 * M1) * iT
        # print(int(((tt + 1) / nt) * 10000) / 100, '% of №', n, ' test', sep='')
        # if c == 0:
        #     c += 1
        #     between = time.time()
        # if tt != range(nt)[-1]:
        #     print('Left:',
        #           int((((int(((between - ttt)) * 1000) / 1000) *
        #                 ((100 / (int((1 / nt) * 10000) / 100)) - (tt + 1))
        #                 ) / 60) * 100) / 100,
        #           'mins')
    # plot the calculated values
    n = 1
    print('Creating plots for №', n, ' test', sep='')
    f = plt.figure(figsize=(18, 10))

    sp1 = f.add_subplot(2, 2, 1)
    plt.scatter(T, E, s=20, marker='o', color='IndianRed')
    plt.xlabel("$T$", fontsize=10)
    plt.ylabel("E", fontsize=10)
    plt.axis('tight')

    sp2 = f.add_subplot(2, 2, 2)
    plt.scatter(T, abs(M), s=20, marker='o', color='RoyalBlue')
    plt.xlabel("$T$", fontsize=10)
    plt.ylabel("M ", fontsize=10)
    plt.axis('tight')

    sp3 = f.add_subplot(2, 2, 3)
    plt.scatter(T, C, s=20, marker='o', color='IndianRed')
    plt.xlabel("$T$", fontsize=10)
    plt.ylabel("$C_v$", fontsize=10)
    plt.axis('tight')

    sp4 = f.add_subplot(2, 2, 4)
    plt.scatter(T, X, s=20, marker='o', color='RoyalBlue')
    plt.xlabel("$T$", fontsize=10)
    plt.ylabel("$\chi$", fontsize=10)
    plt.axis('tight')

    name = 'ising_3d_MC_test' + str(n) + '_N=' + str(N) + '.png'
    plt.savefig(name, bbox_inches='tight', dpi=500)  # plotting
    print('№', n, ' test completed', sep='')
    # print('------')
    # print("%s minutes elapsed" % (
    #         int(((time.time() - start_time) / 60) * 1000) / 1000))  # elapsed time
    print('------')
    # if n == u - 1:
    #     print("Total %s minutes elapsed" % (
    #             int(((time.time() - Start_time) / 60) * 1000) / 1000))  # elapsed time


# break

def processesed(procs, calc):
    processes = []
    for proc in range(procs):
        p = multiprocessing.Process(target=sim_tt, args=(calc, proc, nt, N))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
if __name__ == "__main__":
    n_proc = multiprocessing.cpu_count()
    calc = 500 // n_proc + 1
    start = time.time()
    nt = int(calc * n_proc)  # number of temperature points
    eqSteps = 2 ** 8  # number of MC sweeps for equilibration
    mcSteps = 2 ** 8  # number of MC sweeps for calculation
    N = 5  # size of lattice
    processesed(n_proc, calc)
    end = time.time()
    print('total time:', end - start)
