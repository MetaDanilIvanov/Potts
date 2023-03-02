from scipy.optimize import curve_fit
from matplotlib import pyplot
import multiprocessing
import numpy as np
with open("E.txt", "r") as f:
    E = (eval(f.readline()))
with open("M.txt", "r") as f:
    M = abs(np.array(eval(f.readline()))).tolist()
with open("C.txt", "r") as f:
    C = (eval(f.readline()))
with open("X.txt", "r") as f:
    X = (eval(f.readline()))
with open("T.txt", "r") as f:
    T = (eval(f.readline()))

Ccc = []
Ttt = []
for tp in T:
    if tp > 2:
        Ttt.append(tp)
        Ccc.append(C[T.index(tp)])
Ccc.sort()
t_c = T[C.index(Ccc[-2])]
print(f"t_c = {t_c}")


# i = 0
#
taus = []
Mm = []
# for tp in T:
#     taus.append((tp/t_c)-1)
# tau = taus[taus.index(0.0)-1]
# print(f"tau = {tau}")
for i in range(1, 11):
    taus.append(abs(T[T.index(t_c)-i]-t_c))
    Mm.append(M[T.index(t_c)-i])

print(taus)
print(Mm)

with open("E.txt", "r") as f:
    E = (eval(f.readline()))
with open("M.txt", "r") as f:
    M = abs(np.array(eval(f.readline()))).tolist()
with open("C.txt", "r") as f:
    C = (eval(f.readline()))
with open("X.txt", "r") as f:
    X = (eval(f.readline()))
with open("T.txt", "r") as f:
    T = (eval(f.readline()))

# alpha = np.log(abs(C[taus.index(tau)])) / np.log(abs(tau))
# beta = np.log(abs(M[taus.index(tau)])) / np.log(abs(tau))
# gamma =np.log(abs(X[taus.index(tau)])) / np.log(abs(tau))



def objective(x, a, b):
    return a * x + b

x, y = np.log(taus), np.log(Mm)
popt, _ = curve_fit(objective, x, y)
a, b = popt
pyplot.scatter(x, y)
x_line = np.linspace(min(x), max(x), 2)
y_line = objective(x_line, a, b)
pyplot.plot(x_line, y_line, '--', color='red')
pyplot.show()
print(f"beta = {a}")
beta = a
# print(f"alpha = {alpha}")
print(f"beta = {beta}")
# print(f"gamma = {gamma}")

# print(alpha + 2*beta + gamma)
#
#
# x, y = np.log(Tt), np.log(Xx)
# popt, _ = curve_fit(objective, x, y)
# a, b = popt
# pyplot.scatter(x, y)
# x_line = np.linspace(min(x), max(x), 2)
# y_line = objective(x_line, a, b)
# pyplot.plot(x_line, y_line, '--', color='red')
# pyplot.show()
# print(f"gamma = {a}")
#
# x, y = np.log(Tt), np.log(Cc)
# popt, _ = curve_fit(objective, x, y)
# a, b = popt
# pyplot.scatter(x, y)
# x_line = np.linspace(min(x), max(x), 2)
# y_line = objective(x_line, a, b)
# pyplot.plot(x_line, y_line, '--', color='red')
# pyplot.show()
# print(f"alpha = {a}")
#
# import numpy as np
# from numpy.random import rand
#
# def mcmove(config, beta, B):
#     """Monte Carlo move using Metropolis algorithm"""
#     for a in range(N):
#         for b in range(N):
#             for c in range(N):
#                 i = np.random.randint(0, N)
#                 j = np.random.randint(0, N)
#                 k = np.random.randint(0, N)
#                 s = config[i, j, k]
#                 cost = 2 * s * ((B ** (((i + 1) % N) != (i + 1))) * config[(i + 1) % N, j, k]
#                                 + (B ** (((j + 1) % N) != (j + 1))) * config[i, (j + 1) % N, k]
#                                 + (B ** (((i - 1) % N) != (i - 1))) * config[(i - 1), j, k]
#                                 + (B ** (((j - 1) % N) != (j - 1))) * config[i, (j - 1), k]
#                                 + (B ** (((k + 1) % N) != (k + 1))) * config[i, j, (k + 1) % N]
#                                 + (B ** (((k - 1) % N) != (k - 1))) * config[i, j, (k - 1)])
#                 if cost < 0:
#                     s *= -1
#                 elif rand() < np.exp(-cost * beta):
#                     s *= -1
#                 config[i, j, k] = s
#     return config
#
#
# def calcEnergy(config, B):
#     """Energy of a given configuration"""
#     energy = np.array(np.zeros(1), dtype=np.longdouble)
#     for i in range(N):
#         for j in range(N):
#             for k in range(N):
#                 energy += -((B ** (((i + 1) % N) != (i + 1))) * config[(i + 1) % N, j, k]
#                             + (B ** (((j + 1) % N) != (j + 1))) * config[i, (j + 1) % N, k]
#                             + (B ** (((i - 1) % N) != (i - 1))) * config[(i - 1), j, k]
#                             + (B ** (((j - 1) % N) != (j - 1))) * config[i, (j - 1), k]
#                             + (B ** (((k + 1) % N) != (k + 1))) * config[i, j, (k + 1) % N]
#                             + (B ** (((k - 1) % N) != (k - 1))) * config[i, j, (k - 1)]) * config[i, j, k]
#     return energy / 6.
#
#
#
# eqSteps = 2 ** 9  # number of MC sweeps for equilibration
# mcSteps = 2 ** 9  # number of MC sweeps for calculation
# N = 7  # size of lattice
# b = 1
# imp = 1
# n1, n2 = 1.0 / (mcSteps * N * N * N), 1.0 / (mcSteps * mcSteps * N * N * N)
# iT = 1.0 / t_c+0.00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001
# E1 = np.array(np.zeros(1), dtype=np.longdouble)
# M1 = np.array(np.zeros(1), dtype=np.longdouble)
# E2 = np.array(np.zeros(1), dtype=np.longdouble)
# M2 = np.array(np.zeros(1), dtype=np.longdouble)
# config = (((2 - (imp == 0)) * np.random.randint(3 - imp, size=(N, N, N))) - (imp != 0)) - (
#         (imp == 0) * np.ones((N, N, N)))
# for i in range(eqSteps):  # equilibrate
#     mcmove(config, iT, b)  # Monte Carlo moves
# for i in range(mcSteps):
#     mcmove(config, iT, b)
#     Ene = calcEnergy(config, b)  # calculate the energy
#     Mag = np.sum(config, dtype=np.longdouble)  # calculate the magnetisation
#     E1 += Ene
#     M1 += Mag
#     M2 += (n1 * Mag * Mag)
#     E2 += (n1 * Ene * Ene)
# E_loc, M_loc, C_loc, X_loc = n1 * E1, n1 * M1, (E2 - n2 * E1 * E1) * (iT * iT), (M2 - n2 * M1 * M1) * iT
# beta = np.log(abs(M_loc)) / np.log(abs(tau))
# gamma = np.log(abs(X_loc)) / np.log(abs(tau))
# alpha = np.log(abs(C_loc)) / np.log(abs(tau))
# print(f"beta = {beta}")
# print(f"gamma = {gamma}")
# print(f"alpha = {alpha}")
# print(alpha + 2*beta + gamma)
print(f"табличные a, b, c 0.125±0.015	0.312±0.003	1.250±0.003")