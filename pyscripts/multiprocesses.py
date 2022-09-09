# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 08:59:35 2022

@author: rlsmi
"""

import multiprocessing as mp
import matplotlib.pyplot as plt
import time
import numpy as np
# %matplotlib notebook
print((f'Number of cpu: {mp.cpu_count()}'))


def random_square(seed):
    np.random.seed(seed)
    random_num = np.random.randint(0, 10)
    return random_num**2


to = time.time()
results = []
n = 1000
for i in range(n):
    results.append(random_square(i))
ti = time.time()
print(f'execution time = {ti - to}s')

t0 = time.time()
n_cpu = mp.cpu_count()
print((f'Number of cpu: {n_cpu}'))
n = 1000

if __name__ == '__main__':
    pool = mp.Pool(processes=n_cpu)
    results = [pool.map(random_square, range(1000))]
t1 = time.time()
print(f'execution time = {t1 - t0}s')