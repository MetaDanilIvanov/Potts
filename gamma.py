from matplotlib.ticker import MultipleLocator
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


def func_powerlaw(T, k=1, gamma=-1, Tc=2.268):
    return k * np.abs((T - Tc)/Tc) ** gamma


Tt = []
Xx = []
t_c = T[np.argmax(C)]
t_c = 2.26
for t in T:
    if t >= t_c:
        Tt.append(t)
        Xx.append(X[T.index(t)])
N_T = len(Tt)
Ts = Tt
N = 10
Ks = 0
gammas = 0
sigma_gamma = 0
TCs = 0
col = 'red'
Chi = Xx
i = 0
sigma_TCs = 0
for tt in range(100):
    Tmax = np.argmax(Xx[2:])
    sol, cov = curve_fit(func_powerlaw, Ts[Tmax + tt:], Chi[Tmax + tt:], maxfev=int(1e6))
    if sol[2] > 2.1 and sol[2] < 2.3:
        print(f"γ = {-sol[1]:.3f}, T_c = {sol[2]:.3f}, k = {sol[0]:.3f}")
    # Ks = sol[0]
    # gammas = sol[1]
    # TCs = sol[2]
    # sigma_gamma = cov[1, 1]
    # sigma_TCs = cov[2, 2]
    # ax.plot(Ts[Tmax + tt:], func_powerlaw(Ts[Tmax + tt:], Ks, gammas, TCs), 'orange', label='fit')
    # ax.plot(Ts[Tmax + tt:], Chi[Tmax + tt:], 'o', color=col, label='N={}'.format(N))
    # ax.text(0.8, 0.5, '$\gamma$ = {}\n$T_c$ = {}'.format('%.3f' % (-1 * gammas), '%.3f' % TCs), transform=ax.transAxes,
    #         bbox=dict(facecolor=col, alpha=0.7), fontsize=12)
    # ax.set_xlabel('T', fontsize=16)
    # ax.set_ylabel('smth', fontsize=16)
    # ax.set_title('Fit N={}'.format(N), fontsize=16, fontweight="bold")
    # ax.xaxis.set_minor_locator(MultipleLocator(0.05))
    # ax.yaxis.set_minor_locator(MultipleLocator(1))
    # plt.legend()
    # # print(gammas, TCs)
    # #
    # plt.subplots_adjust(hspace=0)
    # # plt.show()
#
#
#
# # t_c = 4.5
# t_c = 2.268
# taus = []
# taus2 = []
# Mm = []
# Cc = []
# Xx = []
# i = 1
# for t in T:
#     i -= 1
#     if i > -6:
#         pass
#     else:
#         if t - t_c != 0:
#             Cc.append(abs(C[T.index(t)]))
#             taus.append(abs(t-t_c)/t_c)
#             Xx.append(abs(X[T.index(t)]))
#             if t < t_c:
#                 Mm.append(abs(M[T.index(t)]))
#                 taus2.append(abs(t-t_c)/t_c)
# def objective(x, a, b):
#     return a * x + b
#
# x, y = np.log(taus), np.log(Cc)
# popt, _ = curve_fit(objective, x, y)
# a, b = popt
#
# plt.scatter(x, y)
# plt.xlabel("$tau$", fontsize=15)
# plt.ylabel("$C_v$", fontsize=15)
# plt.axis('tight')
# x_line = np.linspace(min(x), max(x), 2)
# y_line = objective(x_line, a, b)
# plt.plot(x_line, y_line, '--', color='red')
# plt.show()
# alpha = -a
# print(f"α = {alpha:.3f}")
# x, y = np.log(taus2), np.log(Mm)
# popt, _ = curve_fit(objective, x, y)
# a, b = popt
# plt.scatter(x, y)
# plt.xlabel("$tau$", fontsize=15)
# plt.ylabel("$M$", fontsize=15)
# plt.axis('tight')
# x_line = np.linspace(min(x), max(x), 2)
# y_line = objective(x_line, a, b)
# plt.plot(x_line, y_line, '--', color='red')
# plt.show()
# beta = a
# print(f"β = {beta:.3f}")
# x, y = np.log(taus), np.log(Xx)
# popt, _ = curve_fit(objective, x, y)
# a, b = popt
# plt.scatter(x, y)
# plt.xlabel("$tau$", fontsize=15)
# plt.ylabel("$\chi$", fontsize=15)
# plt.axis('tight')
# x_line = np.linspace(min(x), max(x), 2)
# y_line = objective(x_line, a, b)
# plt.plot(x_line, y_line, '--', color='red')
# plt.show()
# gamma = -a
# print(f"γ = {gamma:.3f}")
# print(f"α+2β+γ = {alpha + 2*beta + gamma:.3f}")
print(f"2d табличные α, β, γ:  0; 0.125; 1.75")
# print(f"3d табличные α, β, γ: 0.125±0.015; 0.312±0.003; 1.250±0.003")
