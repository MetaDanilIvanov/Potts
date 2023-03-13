import numpy as np
import matplotlib.pyplot as plt

#define data
data = [4, 6, 6, 8, 9, 14, 16, 16, 17, 20]
std_error = np.std(data, ddof=1) / np.sqrt(len(data))
#define x and y coordinates
x = np.arange(len(data))
y = data

#create line chart with error bars
fig, ax = plt.subplots()

ax.errorbar(x, y, yerr=std_error, capsize=4, ecolor='red', elinewidth=0.4)
plt.show()