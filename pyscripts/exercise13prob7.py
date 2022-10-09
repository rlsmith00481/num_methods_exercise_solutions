# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 21:39:50 2022

@author: rlsmi
"""

from joblib import Parallel, delayed



def plus_cube(x, y):
    return (x + y)**3


def parallel_m():
    results = Parallel(n_jobs=-1, backend='multiprocessing', verbose=1)\
        (delayed(plus_cube)(x1, y1)
         for x1, y1 in zip(range(1000), range(1000)))
    print(results)


def main():
    parallel_m()


if __name__ == '__main__':
    print('EX7')
    main()
