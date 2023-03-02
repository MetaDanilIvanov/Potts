import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
import matplotlib
import time
import multiprocessing
import os
import scipy.constants
from datetime import datetime
#
# with open("E.txt", "r") as f:
#     E = (eval(f.readline()))
# with open("M.txt", "r") as f:
#     M = abs(np.array(eval(f.readline()))).tolist()
# with open("C.txt", "r") as f:
#     C = (eval(f.readline()))
# with open("X.txt", "r") as f:
#     X = (eval(f.readline()))
# with open("T.txt", "r") as f:
#     T = (eval(f.readline()))
# C = C[(T.index(2.4481037924207185)):]
# M = M[(T.index(2.4481037924207185)):]
# E = E[(T.index(2.4481037924207185)):]
# X = X[(T.index(2.4481037924207185)):]
# T = T[(T.index(2.4481037924207185)):]
#
# f = plt.figure(figsize=(18, 10))
#
# sp1 = f.add_subplot(2, 2, 1)
# plt.scatter(T, E, s=3, marker='o', color='IndianRed')
# plt.xlabel("$T$", fontsize=25)
# plt.ylabel("$E$", fontsize=25)
# plt.axis('tight')
#
# sp2 = f.add_subplot(2, 2, 2)
# plt.scatter(T, abs(np.array(M)), s=3, marker='o', color='RoyalBlue')
# plt.xlabel("$T$", fontsize=25)
# plt.ylabel("$M$ ", fontsize=25)
# plt.axis('tight')
#
# sp3 = f.add_subplot(2, 2, 3)
# plt.scatter(T, C, s=3, marker='o', color='IndianRed')
# plt.xlabel("$T$", fontsize=25)
# plt.ylabel("$C_v$", fontsize=25)
# plt.axis('tight')
#
# sp4 = f.add_subplot(2, 2, 4)
# plt.scatter(T, X, s=3, marker='o', color='RoyalBlue')
# plt.xlabel("$T$", fontsize=25)
# plt.ylabel("$\chi$", fontsize=25)
# plt.axis('tight')
#
# plt.show()

print(scipy.constants.k)