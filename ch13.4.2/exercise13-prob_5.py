# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 18:44:51 2022

@author: rlsmi
"""

import numpy as np
import multiprocessing as mp
import time
print('EX5')


def random_square(num):
    np.random.seed(num)
    random_num = np.random.randint(0, 10)
    return random_num**2


def parallel(n):
    t0 = time.time()
    n_cpu = mp.cpu_count()
    pool = mp.Pool(processes=n_cpu)
    results = pool.map(random_square, range(n))
    print(f'The results using map: {results}')
    t1 = time.time()
    exec_time = t1 - t0
    return exec_time


def parallel2(n):
    t0 = time.time()
    n_cpu = mp.cpu_count()
    pool = mp.Pool(processes=n_cpu)
    results = pool.map_async(random_square, range(n))
    for results in results.get():
        print(f'The results using map_async: {results}', flush=True)
    t1 = time.time()
    exec_time = t1 - t0
    
    return exec_time


def main(n):
    exec1 = parallel(n)
    exec2 = parallel2(n)
    print(f'Using map:{exec1}\nUsing map-async:{exec2}')

if __name__ == '__main__':
    print('EX5')
    print("""
Map will block until all the work is complete (or an exception is thrown).
Where as map will pass through the exception before all the work is completed.
The built-in map() function allows you to apply a function to each item
in an iterable. The map_async() function does not block while the function
is applied to each item in the iterable, instead it returns a AsyncResult
object from which the results may be accessed. It returns a AsyncResult
Object. To get results from the function map_async a results.get() is needed
to return the values.""")
    main(10)
