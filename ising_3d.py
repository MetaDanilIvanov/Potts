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
    Eeee = [[], []]
    Mmmm = [[], []]
    Cccc = [[], []]
    Xxxx = [[], []]
    for i in range(calc):
        lists = sim_tt(N, (calc * proc + i))
        Eeee[i] = sum(lists[0])
        Mmmm[i] = sum(lists[1])
        Cccc[i] = sum(lists[2])
        Xxxx[i] = sum(lists[3])
    file = open("E_%s.txt" % proc, "w")
    file.write(str(Eeee))
    file.close()
    file = open("M_%s.txt" % proc, "w")
    file.write(str(Mmmm))
    file.close()
    file = open("C_%s.txt" % proc, "w")
    file.write(str(Cccc))
    file.close()
    file = open("X_%s.txt" % proc, "w")
    file.write(str(Xxxx))
    file.close()
def sim_tt(N, tt):
    n1, n2 = 1.0 / (mcSteps * N * N), 1.0 / (mcSteps * mcSteps * N * N)
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
it = 16
calc = it // n_proc + ((it // n_proc) != (it / n_proc))
nt = int(calc * n_proc)  # number of temperature points
eqSteps = 2 ** 5  # number of MC sweeps for equilibration
mcSteps = 2 ** 5  # number of MC sweeps for calculation
N = 5  # size of lattice
T = np.linspace(2., 7., nt)  # 4.5
E, M, C, X = np.array(np.zeros(nt), dtype=np.float64), np.array(np.zeros(nt), dtype=np.float64), \
             np.array(np.zeros(nt), dtype=np.float64), np.array(np.zeros(nt), dtype=np.float64)
if __name__ == "__main__":
    processesed(n_proc, calc)





