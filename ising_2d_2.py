import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
import matplotlib
import time
import multiprocessing
import os
import scipy.constants
from datetime import datetime

matplotlib.rcParams['text.latex.preamble'] = \
    r'\usepackage{amsmath} \usepackage{amssymb} \usepackage{palatino} \usepackage{textcomp}'
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Tahoma']

# 2d
n_proc = multiprocessing.cpu_count()
it = 48
calc = it // n_proc + ((it // n_proc) != (it / n_proc))
nt = int(calc * n_proc)  # number of temperature points
N = 18  # size of lattice
T = np.linspace(2.255, 2.265, nt)  # 2.26
H = 0

def beta(T):
    return (scipy.constants.k*T)**-1
print(beta(1.e23))

J = 1