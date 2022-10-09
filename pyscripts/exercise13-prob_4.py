# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 16:12:33 2022

@author: rlsmi
"""

import time
import multiprocessing as mp
print('EX4')


def plus_cube(x, y):
    return (x + y)**3


def cube(x1, y1, n):
    to = time.time()
    results = []
    for x1, y1 in zip(range(n), range(n)):
        results.append(plus_cube(x1, y1))
    ti = time.time()
    exec_time = ti - to
    return exec_time


def cube_m(x2, y2, n):
    results = []
    t0 = time.time()
    n_cpu = mp.cpu_count()
    print((f'Number of cpu: {n_cpu}'))
    pool = mp.Pool(processes=n_cpu)

    results = [pool.apply_async(plus_cube(x2, y2), range(n), range(n))]
    t1 = time.time()
    exec_time = t1 - t0
    return exec_time


def main(x, y, num):
    exec_time_ser = cube(x, y, num)
    exec_time_par = cube_m(x, y, num)
    return exec_time_ser, exec_time_par


if __name__ == '__main__':
    xf = 25
    yf = 25
    nf = 1000000
    excution_time_s, excution_time_p =main(25, 25, 1000000)
    print(f'Serial execution takes {excution_time_s} s'
          f' and Parallel is {excution_time_p} s'
          f' for x = {xf}, y = {yf}, number of iterations = {nf}')
    print(f'The difference in the time of excution for between'
          f' serial and parallel is {excution_time_s - excution_time_p} s')

