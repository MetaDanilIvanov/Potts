import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
import copy
import time


# завёрнутая модель
# нет внешнего поля
# 3d
def initialstate(N: int):
    """generates a random spin configuration for initial condition 3d"""
    state = []
    for levels in range(N):
        slicee = 2 * np.random.randint(2, size=(N, N)) - 1
        state.append(slicee.tolist())
    return np.array(state)


def mcmove(config, beta):
    """Monte Carlo move using Metropolis algorithm """
    for a in range(N):
        for b in range(N):
            for c in range(N):
                i = np.random.randint(0, N)
                j = np.random.randint(0, N)
                k = np.random.randint(0, N)
                s = config[i, j, k]
                config2 = copy.deepcopy(config)
                config2[i, j, k] *= -1
                de = calcEnergy(config2) - calcEnergy(config)
                if de < 0:
                    s *= -1
                elif rand() < np.exp(-de * beta):
                    s *= -1
                config[a, b, c] = s
    return config


def calcEnergy(config):
    """Energy of a given configuration"""
    energy = 0
    for i in range(len(config)):
        for j in range(len(config)):
            for k in range(len(config)):
                nb = config[(i - 1) % N, j] * config[i, j] + \
                     config[i, j] * config[(i + 1) % N, j] + \
                     config[i, (j - 1) % N] * config[i, j] + \
                     config[i, j] * config[i, (j + 1) % N] + \
                     config[(i - 1) % N, k] * config[i, k] + \
                     config[i, k] * config[(i + 1) % N, k] + \
                     config[i, (k - 1) % N] * config[i, k] + \
                     config[i, k] * config[i, (k + 1) % N] + \
                     config[(j - 1) % N, k] * config[j, k] + \
                     config[j, k] * config[(j + 1) % N, k] + \
                     config[j, k] * config[j, (k - 1) % N] + \
                     config[j, k] * config[j, (k + 1) % N]
                energy += sum((-J * nb * config[i, j, k]) / 12.)
    return energy


def calcMag(config):
    """Magnetization of a given configuration"""
    mag = np.sum(config)
    return mag


# u = int(input('How much simulations do you need?\t'))
u = 3
for n in range(1, u):
    start_time = time.time()
    nt = 20  # number of temperature points
    N = 10  # size of the lattice, N x N
    eqSteps = 5 * (10 ** 0)  # number of MC sweeps for equilibration
    mcSteps = 1 * (10 ** 0)  # number of MC sweeps for calculation
    J = 1  # interaction
    # no external field

    T = np.linspace(3., 6., nt)  # 4.5
    E, M, C, X = np.array(np.zeros(nt), dtype=np.float64), np.array(np.zeros(nt), dtype=np.float64), np.array(
        np.zeros(nt), dtype=np.float64), np.array(np.zeros(nt), dtype=np.float64)
    n1, n2 = 1.0 / (mcSteps * N * N), 1.0 / (mcSteps * mcSteps * N * N)
    ttt = time.time()
    c = 0
    between = 0
    for tt in range(nt):
        E1 = np.array(np.zeros(1), dtype=np.float64)
        M1 = np.array(np.zeros(1), dtype=np.float64)
        E2 = np.array(np.zeros(1), dtype=np.float64)
        M2 = np.array(np.zeros(1), dtype=np.float64)  ###
        config = initialstate(N)
        iT = 1.0 / T[tt]

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
        X[tt] = (n1 * M2 - n2 * M1 * M1) * iT * J
        print(int(((tt + 1) / nt) * 10000) / 100, '% of №', n, ' test', sep='')
        if c == 0:
            c += 1
            between = time.time()
        print('Left:',
              int((((int(((between - ttt)) * 1000) / 1000) *
                    ((100 / (int((1 / nt) * 10000) / 100)) - (tt + 1))
                    ) / 60) * 100) / 100,
              'mins')
    # plot the calculated values
    f = plt.figure(figsize=(18, 10))

    sp1 = f.add_subplot(2, 2, 1)
    plt.scatter(T, E, s=50, marker='o', color='IndianRed')
    plt.xlabel("Temperature (T)", fontsize=20)
    plt.ylabel("Energy ", fontsize=20)
    plt.axis('tight')

    sp2 = f.add_subplot(2, 2, 2)
    plt.scatter(T, abs(M), s=50, marker='o', color='RoyalBlue')
    plt.xlabel("Temperature (T)", fontsize=20)
    plt.ylabel("Magnetization ", fontsize=20)
    plt.axis('tight')

    sp3 = f.add_subplot(2, 2, 3)
    plt.scatter(T, C, s=50, marker='o', color='IndianRed')
    plt.xlabel("Temperature (T)", fontsize=20)
    plt.ylabel("Specific Heat ", fontsize=20)
    plt.axis('tight')

    sp4 = f.add_subplot(2, 2, 4)
    plt.scatter(T, X, s=50, marker='o', color='RoyalBlue')
    plt.xlabel("Temperature (T)", fontsize=20)
    plt.ylabel("Susceptibility", fontsize=20)
    plt.axis('tight')

    name = 'ising_3d_MC_test' + str(n) + '.png'
    plt.savefig(name, bbox_inches='tight', dpi=800)  # plotting
    print('№', n, ' test completed', sep='')
    if n == 1:
        print("--- %s minutes per test---" % (
                int(((time.time() - start_time) / 60) * 1000) / 1000))  # timing of each test
    break
