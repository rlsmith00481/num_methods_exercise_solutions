# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 09:41:10 2022

@author: rlsmi
"""
import numpy as np
class random_seed:
    def __init__(self, n=0):
        self.num = n


    def random_square(self, num):
        np.random.seed(num)
        self.random_num = np.random.randint(0, 10)
        return self.random_num**2

