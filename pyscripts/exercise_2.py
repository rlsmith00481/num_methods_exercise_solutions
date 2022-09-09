#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from math import*
import re
print('2.8.2 Problems Python Programming and Numerical Methods')
# 'EX 1, 2, 3'
try:
    y = x1**2
except Exception as e:
    print('Create the error code; NameError:', e)

x, y = 2, 3
print(f'Assign x and y, x = {x} and y = {y}\n')


def assignments(x, y):
    u = x + y
    v = x * y
    w = x / y
    z = sin(x)
    r = 8 * sin(x)
    s = 5 * sin(x*y)
    p = x**y
    return [u, v, w, z, r, s, p]


ass_lst = ['u', 'v', 'w', 'z', 'r', 's', 'p']
opr_lst = np.around(assignments(10, 3), 3)

df_ass = pd.DataFrame({'Variables': ass_lst, 'Values':  opr_lst},
                      index=('x + y', 'xy', 'x/y', 'sin(x)', '8sin(x)',
                             '5sin(xy)', 'x**y'),
                      columns=['Variables', 'Values'])
df_ass.index.name = "Expressions"
print('Showing all variables and values from the given expressions.')
print(df_ass, '\n')

# EX 5, 6,  8, 9
S = '123'
N = float(S)
print('S is of number type', type(S), S)
print('N is of number type', type(N), N, '\n')

s1 = 'HELLO'
s2 = 'hello'
print(f'The upper case {s1} and lower case {s2} words are not equal;'
      ' is {s1} equal to {s2}, {s1 == s2}')
print(f'Putting {s1} to lower case {s1.lower()} the words are then equal;'
      ' is {s1.lower()} equal to {s2}, {s1.lower() == s2}')
print(f'Changing {s2} to upper case {s2.upper()} the words are then equal;'
      ' is {s2.upper()} equal to {s1}, {s1 == s2.upper()}\n')

w_1 = int(len('Engineering'))
w_lst = list('Engineering')
w_2 = int(len('Book'))
w_2st = list('Book')
tot = w_1 + w_2
print(f'The count of the letter in Engineering ({w_lst}) is {w_1} and the'
      f' letter count in Book ({w_2st}) is {w_2} '
      f'and total letter count is {tot}.\n')

words = 'Python is Great'

word1 = re.search(r"(^\w+)", words)
# Start the match at the begainning ^ matching all charaters\w+ until a space
word2 = re.search(r"(\w+)$", words)
# Start the match at the end $ matching all charaters\w+ until a space
print(f'Using the re.match exact pattern, the word Python was found'
      f' in "{words}"')
print(word1.group(0), '\n')
# Group zero which means every thing in all groups.
print(f'Using a general regex pattern a search for the last word was'
      f' completed, the last word Great was found in "{words}"')
print(word2.group(0), '\n')

# EX 10, 11, 12, 13, 14

list_a = [1, 8, 9, 15]
print(f'The orginal list is: {list_a}.')
list_a[1] = 2
print(f'Insert 2 at index number 1 in orginal list: {list_a}.')
list_a.append(4)
print(f'Append the orginal list with a 4: {list_a}.')
list_a.sort()
print(f'This is the sort list: {list_a}.\n')

words_list = list(words)
print(f'This is a list of, "{words}".')
print(words_list, '\n')

tuple_a = ('One', 1)
print(f'This is the variables assiged to tuple_a {tuple_a}.')
print('Get the second element in tuple_a', tuple_a[1:3], '\n')

# EX 15, 16

set_o = {2, 3, 2, 3, 1, 2, 5}
print('Unique elements of (2, 3, 2, 3, 1, 2, 5) are:', set_o)
set_a = {2, 3, 2}
set_b = {1, 2, 3}
set_c = {11, 2, 8}
set_u = set_a.union(set_b)
print('The union of (2, 3, 2) and (1, 2, 3) is:', set_u)
set_i = set_a.intersection(set_b)
print('The intersection of (2, 3, 2) and (1, 2, 3) is:', set_i)
set_s = set_a.issubset(set_b)
print('Is (2, 3, 2) a sub set of (1, 2, 3)?', set_s)
set_d = set_c.difference(set_a)
print('The difference between  (11, 3, 8) and (2, 3, 2) is:', set_d, '\n')

# EX 17, 18

keys = ['A', 'B', 'C']
val = ['a', 'b', 'c']
dict_1 = dict(zip(keys, val))
dict_keys = dict_1.keys()

print(f'The keys for the dictionary {dict_1} are {keys}.')
print('The value of key "B" is:', dict_1['B'], '\n')

# EX 19, 20, 21, 22, 24

x = np.array([1, 4, 3, 2, 9, 4])
y = np.array([2, 3, 4, 1, 2, 3])


def assignments(x, y):
    u = np.add(x, y)
    v = np.multiply(x, y)
    w = np.divide(x, y)
    z = np.sin(x)
    r = 8 * np.sin(x)
    s = 5 * np.sin(np.multiply(x, y))
    p = np.power(x, y)
    return [u, v, w, z, r, s, p]


ass_lst = ['u', 'v', 'w', 'z', 'r', 's', 'p']
opr_lst = np.around(assignments(x, y), 3)
opr_lst = list(opr_lst)
df_ass = pd.DataFrame({'Arrays': ass_lst, 'Array Values':  opr_lst},
                      index=('x + y', 'xy', 'x/y', 'sin(x)',
                             '8sin(x)', '5sin(xy)', 'x**y'),
                      columns=['Arrays', 'Array Values'])
df_ass.index.name = "Expressions"
print('Showing all variables and values from the given expressions.')
print(f'Where x is an array of {x} and y is an array of {y}')
print(df_ass, '\n')

arr_100 = np.linspace(-10, 10, 100)
arr_100 = np.around(arr_100, 5)
print(f'This is an array of 100 numbers that are evenly spaced'
      f' between -10 and 10.\n {arr_100}')
print('The number of elemnets in the array is:', arr_100.shape, '\n')

array_a = np.array([-1, 0, 1, 2, 0, 3])
print(f'All elements that are greater than 0.0 in {array_a} are'
      f' {array_a[array_a > 0]}.')

array_y = np.array([[3, 5, 3], [2, 2, 5], [3, 8, 9]])
print(f'The orginal 3x3 matrix \n {array_y}')
print(f'The transposed 3x3 matrix \n {array_y.T}\n')

dim = (2, 4)
array_24 = np.zeros(dim)
print(f'This is a 2x4 zero array\n {array_24}.\n')
array_24[:, 1] = 1.0
print(f'The zero array column 1 was changed to 1.0\n {array_24}\n')

# EX 25


all_var = get_ipython().run_line_magic('who', '')
print('This is all the variables used in chapter 2 problems')
print(all_var)
get_ipython().run_line_magic('reset', '')
print("This is what what remained after the varible reset.")
all_var = get_ipython().run_line_magic('who', '')
print(all_var)

