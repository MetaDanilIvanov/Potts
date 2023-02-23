# metropolis algorithm for the ising model
import math

import numpy as np
import matplotlib.pyplot as plt
import random
import copy
from matplotlib.animation import ArtistAnimation

# size of lattice (N x M)
N = 6
M = 6
T = 1000  # time steps
J = 1.e50  # interaction
Tm = 2.e73  # temperature
mu = 10  # magnetic moment
k = 1.380649 / 1e23  # boltzmann const
beta = (k * Tm) ** (-1)
s = np.random.choice([-1, 1], size=(M, N))  # conditions
Mu = np.random.choice([-1, 1], size=(M, N))  # magnetic moment
Jj = np.random.choice([-1, 1], size=(M, N))  # interaction
layers = [s.tolist()]


def H(s):  # вычисление гамильтониана
    return -J * (np.sum(s[:-1, :] * s[1:, :]) + np.sum(s[:, :-1] * s[:, 1:]))


def dH(s, i, j):
    S = copy.deepcopy(s)
    S[i, j] *= -1
    H_old = H(s)
    H_new = H(S)
    if H_old > H_new:
        prob = 0
    else:
        prob = H_new - H_old
    return prob


def update(s, T, layers):  # апдейт системы ферромагнетиков
    for t in range(T):
        i, j = np.random.randint(M), np.random.randint(N)
        prob = dH(s, i, j)
        if prob == 0:
            pass
        else:
            probab = math.exp(-beta * prob)
            # print(probab)
            s[i, j] *= random.choices([-1, 1], weights=[probab, 1 - probab])[0]
        layers.append(s.tolist())
    return s, layers


s = update(s, T, layers)  # запуск кода

# ниже -- создание анимации из всех изменений
frames = []
fig = plt.figure()
ax = fig.add_subplot(111)
for t in range(T + 1):
    line = plt.imshow(layers[t], cmap='binary')
    frames.append([line])
animation = ArtistAnimation(fig, frames, interval=0.1, blit=True, repeat=True)

plt.show()