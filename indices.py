from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.rcParams['text.latex.preamble'] = \
    r'\usepackage{amsmath} \usepackage{amssymb} \usepackage{palatino} \usepackage{textcomp}'
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Tahoma']
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


# t_c = 4.5
t_c = 2.268
taus = []
taus2 = []
Mm = []
Cc = []
Xx = []
for t in T:
    if t - t_c != 0:
        Cc.append(abs(C[T.index(t)])/t_c)
        taus.append(abs(t-t_c)/t_c)
        Xx.append(abs(X[T.index(t)])/t_c)
        if t < t_c:
            Mm.append(abs(M[T.index(t)])/t_c)
            taus2.append(abs(t-t_c)/t_c)

def objective(x, a, b):
    return a * x + b

x, y = np.log(taus), np.log(Cc)
popt, _ = curve_fit(objective, x, y)
a, b = popt

plt.scatter(x, y)
plt.xlabel("$tau$", fontsize=15)
plt.ylabel("$C_v$", fontsize=15)
plt.axis('tight')
x_line = np.linspace(min(x), max(x), 2)
y_line = objective(x_line, a, b)
plt.plot(x_line, y_line, '--', color='red')
plt.show()
alpha = -a
print(f"α = {alpha:.3f}")
x, y = np.log(taus2), np.log(Mm)
popt, _ = curve_fit(objective, x, y)
a, b = popt
plt.scatter(x, y)
plt.xlabel("$tau$", fontsize=15)
plt.ylabel("$M$", fontsize=15)
plt.axis('tight')
x_line = np.linspace(min(x), max(x), 2)
y_line = objective(x_line, a, b)
plt.plot(x_line, y_line, '--', color='red')
plt.show()
beta = a
print(f"β = {beta:.3f}")
x, y = np.log(taus), np.log(Xx)
popt, _ = curve_fit(objective, x, y)
a, b = popt
plt.scatter(x, y)
plt.xlabel("$tau$", fontsize=15)
plt.ylabel("$\chi$", fontsize=15)
plt.axis('tight')
x_line = np.linspace(min(x), max(x), 2)
y_line = objective(x_line, a, b)
plt.plot(x_line, y_line, '--', color='red')
plt.show()
gamma = -a
print(f"γ = {gamma:.3f}")
print(f"α+2β+γ = {alpha + 2*beta + gamma:.3f}")
print(f"2d табличные α, β, γ:  0; 0.125; 1.75")
print(f"3d табличные α, β, γ: 0.125±0.015; 0.312±0.003; 1.250±0.003")
