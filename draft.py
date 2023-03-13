import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
import matplotlib
import time
import multiprocessing
import os
from datetime import datetime
N = 5
config = np.random.choice(range(5), size=(N, N))
B = 1
print(config)
for a in range(N):
    for b in range(N):
        i, j = np.random.randint(0, N), np.random.randint(0, N)
        nb = ((B ** (((i + 1) % N) != (i + 1))) * config[(i + 1) % N, j]
              + (B ** (((j + 1) % N) != (j + 1))) * config[i, (j + 1) % N]
              + (B ** (((i - 1) % N) != (i - 1))) * config[(i - 1)%N, j]
              + (B ** (((j - 1) % N) != (j - 1))) * config[i, (j - 1)%N])
        print(config[(i - 1)%N, j], i, j, '1')
        print(config[i, (j - 1)%N], i, j, '2')
