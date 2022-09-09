# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 09:41:10 2022

@author: rlsmi
"""
import numpy as np
import multiprocessing as mp
import time 

def random_square(num):
    np.random.seed(num)
    random_num = np.random.randint(0, 10)
    return random_num**2


print('Done')


def timer(n):
    to = time.time()
    results = []

    for i in range(n):
        results.append(random_square(i))
    ti = time.time()
    print(f'execution time1 = {ti - to}s')

def main(n):
    timer(n)
    to = time.time()
    n_cpu = mp.cpu_count()
    print((f'Number of cpu: {n_cpu}'))
    pool = mp.Pool(processes=n_cpu)

    results = [pool.map(random_square, range(n))]
    ti = time.time()
    print(f'execution time2 = {(ti - to)/1}s')
    
if __name__ == '__main__':
    
    main(10000000)
