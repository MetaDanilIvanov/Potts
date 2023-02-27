import multiprocessing
import time
import numpy as np

def heavy(n, i, proc):
    for x in range(1, n):
        for y in range(1, n):
            x ** y
    print(f"Цикл № {i} ядро {proc}")


def sequential(calc, proc):
    print(f"Запускаем поток № {proc}")
    for i in range(calc):
        heavy(500, i, proc)
    print(f"{calc} циклов вычислений закончены. Процессор № {proc}")

def processesed(procs, calc):
    # procs - количество ядер
    # calc - количество операций на ядро

    processes = []

    # делим вычисления на количество ядер
    for proc in range(procs):
        p = multiprocessing.Process(target=sequential, args=(calc, proc))
        processes.append(p)
        p.start()

    # Ждем, пока все ядра
    # завершат свою работу.
    for p in processes:
        p.join()


# if __name__ == "__main__":
#     start = time.time()
#     # узнаем количество ядер у процессора
#     n_proc = multiprocessing.cpu_count()
#     # вычисляем сколько циклов вычислений будет приходится
#     # на 1 ядро, что бы в сумме получилось 80 или чуть больше
#     calc = 80 // n_proc + 1
#     processesed(n_proc, calc)
#     end = time.time()
#     print(f"Всего {n_proc} ядер в процессоре")
#     print(f"На каждом ядре произведено {calc} циклов вычислений")
#     print(f"Итого {n_proc * calc} циклов за: ", end - start)
calc = 500 // multiprocessing.cpu_count() + 1
n_proc = multiprocessing.cpu_count()

# print(np.linspace(calc, calc * multiprocessing.cpu_count(), multiprocessing.cpu_count()))
# for i in range(multiprocessing.cpu_count()):
#     print(np.linspace(calc*i+1, calc*(i+1), calc))

T = []
for i in range(n_proc):
    T = np.append(T, np.linspace(calc * i + 1, calc * (i + 1), calc))
print(T)