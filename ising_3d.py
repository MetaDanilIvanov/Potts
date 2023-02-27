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


def ising_3d(calc, proc):
    if proc == 0:
        start = time.time()
    Eeee = np.zeros(calc)
    Mmmm = np.zeros(calc)
    Cccc = np.zeros(calc)
    Xxxx = np.zeros(calc)
    for i in range(calc):
        lists = sim_tt(N, (calc * proc + i))
        Eeee[i] = sum(lists[0])
        Mmmm[i] = sum(lists[1])
        Cccc[i] = sum(lists[2])
        Xxxx[i] = sum(lists[3])
        if i == 0 and proc == 0:
            print(int(((((((time.time() - start)))*
                   ((100 / (int((1 / calc) * 10000) / 100)) - 1) / 60) * 100) / 100)*1000)/1000,
                  'minutes left')

    file = open("E_%s.txt" % proc, "w")
    file.write(str(Eeee.tolist()))
    file.close()
    file = open("M_%s.txt" % proc, "w")
    file.write(str(Mmmm.tolist()))
    file.close()
    file = open("C_%s.txt" % proc, "w")
    file.write(str(Cccc.tolist()))
    file.close()
    file = open("X_%s.txt" % proc, "w")
    file.write(str(Xxxx.tolist()))
    file.close()



def sim_tt(N, tt):
    n1, n2 = 1.0 / (mcSteps * N * N), 1.0 / (mcSteps * mcSteps * N * N)
    iT = 1.0 / T[tt]
    E1 = np.array(np.zeros(1), dtype=np.float64)
    M1 = np.array(np.zeros(1), dtype=np.float64)
    E2 = np.array(np.zeros(1), dtype=np.float64)
    M2 = np.array(np.zeros(1), dtype=np.longdouble)
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
    Ee = n1 * E1
    Mm = n1 * M1
    Cc = (n1 * E2 - n2 * E1 * E1) * (iT * iT)
    Xx = (n1 * M2 - n2 * M1 * M1) * iT
    return Ee, Mm, Cc, Xx


def processesed(procs, calc):
    processes = []
    for proc in range(procs):
        p = multiprocessing.Process(target=ising_3d, args=(calc, proc))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()


n_proc = multiprocessing.cpu_count()
it = 200
calc = it // n_proc + ((it // n_proc) != (it / n_proc))
nt = int(calc * n_proc)  # number of temperature points
eqSteps = 2 ** 8  # number of MC sweeps for equilibration
mcSteps = 2 ** 8  # number of MC sweeps for calculation
N = 20  # size of lattice
T = np.linspace(2.3, 6., nt)  # 4.5
if __name__ == "__main__":
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
        E_list.append("E_%s.txt" % i)
        M_list.append("M_%s.txt" % i)
        C_list.append("C_%s.txt" % i)
        X_list.append("X_%s.txt" % i)
    for i in range(n_proc):
        with open(str(E_list[i]), "r") as f:
            E.append(eval(f.readline()))
        with open(str(M_list[i]), "r") as f:
            M.append(eval(f.readline()))
        with open(str(C_list[i]), "r") as f:
            C.append(eval(f.readline()))
        with open(str(X_list[i]), "r") as f:
            X.append(eval(f.readline()))
    E = [a for b in E for a in b]
    M = np.array([a for b in M for a in b])
    C = [a for b in C for a in b]
    X = [a for b in X for a in b]

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

    name = 'ising_3d_MC_test_N=' + str(N) + '.png'
    plt.savefig(name, bbox_inches='tight', dpi=500)