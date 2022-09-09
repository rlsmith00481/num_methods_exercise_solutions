#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 13:43:59 2022

@author: rlsmi
"""

import math
from math import*
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
# EX1


def my_sinh(x):
    """ The function uses "(exp(x) - exp(-x))/2 to calculate the
    hyperbolic sine. The input value is in radians, to convert from
    degrees to radians degree * pi()/180."""
    return (exp(x) - exp(-x))/2


df_sinh = pd.DataFrame({'sinh(0)': my_sinh(0), 'sinh(1)': my_sinh(1),
                        'sinh(2)': my_sinh(2)}, index=(['Values']))
print(f'These are the values at for the hyperbolic sine at 0, 1, 2.\n'
      ' {df_sinh}\n')

# EX2


def my_checker_board(n):
    """
    This function will create a checkerboard square matrix of 0 and 1.
    Enter the size of the matrix n from 1 to 5.
    """
    z = np.ones((n), dtype=float)
    x = np.ones((n), dtype=float)
    np.place(z[1::2], z[1::2] == 1, [0])
    np.place(x[::2], x[::2] == 1, [0])
    if n == 1:
        m = [z]
    elif n == 2:
        m = np.concatenate(([[z], [x]]), axis=0)
    elif n == 3:
        m = np.concatenate(([[z], [x], [z]]), axis=0)
    elif n == 4:
        m = np.concatenate(([[z], [x], [z], [x]]), axis=0)
    elif n == 5:
        m = np.concatenate(([[z], [x], [z], [x], [z]]), axis=0)
    return (m)


print('At n = 1')
print(my_checker_board(1), '\n')
print('At n = 2')
print(my_checker_board(2), '\n')
print('At n = 3')
print(my_checker_board(3), '\n')
print('At n = 4')
print(my_checker_board(4), '\n')
print('At n = 5')
print(my_checker_board(5), '\n')

# EX3

my_triangle = lambda b, h: (b*h)/2
print('The function my_triangle(b, h) calculates the area of a triangle '
      ' using the lambda function.'
      ' Enter the base (b) and height (h) of the triangle into the function.'
      ' The equation for the area of a triangle is 1/2 x base x height or '
      ' 1/2 the area of a rectangle.')
pts = np.array([[0, 0], [1, 0], [0, 1]])
p = Polygon(pts, closed=False)
ax = plt.gca()
ax.add_patch(p)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
plt.show()
print('For the above triangle with a base of 1 and a hieght of 1 '
      f'the area is {my_triangle(1, 1)}.')
pts = np.array([[0, 0], [2, 0], [0, 1]])
p = Polygon(pts, closed=False)
ax = plt.gca()
ax.add_patch(p)
ax.set_xlim(0, 2)
ax.set_ylim(0, 1)
plt.show()
print(f'For the above triangle with a base of 2 and a hieght of 1 '
      f'the area is {my_triangle(2, 1)}.')
pts = np.array([[0, 0], [12, 0], [0, 5]])
p = Polygon(pts, closed=False)
ax = plt.gca()
ax.add_patch(p)
ax.set_xlim(0, 12)
ax.set_ylim(0, 5)
plt.show()
print('For the above triangle with a base of 12 and a hieght of 15'
      f' the area is {my_triangle(12, 5)}.\n')


# EX4


def split_(lst):
    """ Insert a list for the argument in split(lst) in the form of
    ([1,2,3], [4,5,6], [7, 8, 9]). The function will create an array
    and split the array in 2 equal parts. The two parts are retured with
    the function in the form m1, m2 = split(list). Must have at least
    two list objects ([object], [object]) with the same number of elements."""
    m = np.array(lst)
    m = np.array_split(m, 2, axis=1)
    m_1 = m[0]
    m_2 = m[1]
    return m_1, m_2


x = ([1, 2, 3], [4, 5, 6], [7, 8, 9])
m1, m2 = split_(x)
print('For case where array is a ([1, 2, 3], [4, 5, 6], [7, 8, 9])')
print('m1')
print(m1)
print('m2')
print(m2, '\n')

x = np.ones((5, 5))
m1, m2 = split_(x)
print('For case where array is a np.one((5, 5))')
print('m1')
print(m1)
print('m2')
print(m2, '\n')


# EX5


def my_cylinder(r, h):
    """ my_cylinder(r, h) function will calculate the surface area
    and voulume of a cylinder.
    The formula for surface area is 2π(r^2)) + (2πrh) and the volume
    is πr^2h, where r is the radius and h is the height."""
    s = (2 * pi * (r**2)) + (2 * pi * r * h)
    v = pi * (r**2) * h
    return s, v


dict_cly_suf_vol = {'Radius=1 & Height=5':  my_cylinder(1, 5),
                    'Radius=2 & Height=4': my_cylinder(2, 4),
                    'Radius=8.5 & Height=43': my_cylinder(5, 43)}
cly_surface_vol = pd.DataFrame(dict_cly_suf_vol,
                               index=['Cylinder Surface Area',
                                      'Cylinder Volume'])
print('The surface area and volume of a cylinder.')
print(cly_surface_vol, '\n')


# EX6


def my_n_odd(arr):
    """This function does the same thing as the lower function with
    an embeded lambda function using filtering with list. The
    function finds all the odd numbers in the array.
    The function requres a array."""
    odds = list(filter(lambda x: (x % 2 != 0), list(arr)))
    return len(odds)


def my_n_odd2(arr):
    """This function does the same thing as the upper function
    with list comprehension and will work on arrays, lists, or tuple.
    The function finds all the odd numbers in the array.
    The function requres a list of numbers be it a list or array."""
    odds = [x for x in arr if x % 2 != 0]
    return len(odds)


print('The number of odd numbers in the array'
      f' {np.arange(100)} are {my_n_odd(np.arange(100))}.\n')
print('The number of odd numbers in the array'
      f' {np.arange(2, 100, 2)} are'
      f' {my_n_odd(np.arange(2, 100, 2))}.\n')

# EX7


def my_twos(m, n):
    """This function creates a 2D array filled with 2s.
    The function my_twos(m,n) uses (m,n) as the array size. """
    return np.full(shape=(m, n), fill_value=2, dtype=int)


print('This function creates a 2D array filled with 2.')
case_1 = my_twos(3, 2)
print('case 1 is a 2D array with a size of (3 x 2).')
print(case_1, '\n')
case_2 = my_twos(1, 4)
print('case 2 is a 2D array with a size of (1 x 4).')
print(case_2, '\n')

# EX8

sub_xy = lambda x, y: x - y
x, y = 6, 3
print(f'The lambda function for x - y where x = {x} and y = {y},'
      ' so x - y = {sub_xy(x,y)}.\n')


# EX9


def add_string(s_1, s_2=''):
    """Joins two string objects and if they are not string objects
    the function will convert them to strings prior to concatenating
    the strings."""
    s_1 = str(s_1)
    s_2 = str(s_2)
    return s_1 + s_2


s1 = add_string("Programing ")
s2 = add_string("is ", "fun!")
sent = add_string(s1, s2)
print("Case one join 'Programing' and 'is fun!' with function"
      " add_string(s_1, s_2 = '')")
print(sent, '\n')

# EX10


def fun(a):
    print('This is fun')


try:
    err = fun()
except Exception as e:
    print('Create the error code; TypeError:', e, '\n')


def fun():
    f = print('This is fun')
    return print(f)


try:
    err = fun()
except Exception as e:
    print('Create the error code; IndentationError:', e)

# EX11


def greeting(name, age):
    name = str(name)
    age = float(age)
    return print(f'Hi, my names is {name} and I am {age} years old.')


print("Case 1 use 'John' as name and 'age' is 26 to produce the sentence"
      " 'Hi, my name is John and I am 26 years old'.")
greeting('John', 26)
print('\n')
print("Case 2 use 'Kate' as name and 'age' is 19 to produce the sentence"
      " 'Hi, my name is Kate and I am 19 years old'.")
greeting('Kate', 19)
print('\n')


# EX12


def my_donut_area(r1, r2):
    """The function that will calculate the area of a donut with vectorization.
    The function test to make sure the inter radius is lessthan outer radius
    then caculates the area of the inter cut out and subtracts that from total
    area of the circle. r1 is the inter radius and r2 is the outer radius
    where r2 > r1. my_donut_area(r1, r2)"""
    if (r1 < r2).all():
        area_in = np.multiply(pi, np.multiply(r1, r1))
        area_out = np.multiply(pi, np.multiply(r2, r2))
        return np.subtract(area_out, area_in)
    else:
        return 'The inter radius is larger than the outer radius.'


ID = np.arange(1, 4)
OD = np.arange(2, 7, 2)
arr_do = my_donut_area(ID, OD)
print("Test case 2 create the function that will calculate"
      " the area of a donut with vectorization."
      " For inter radius vector [1, 2, 3] & outer radius vector [2, 4, 6].\n")
df_donut = pd.DataFrame(list(zip(OD, ID, arr_do)),
                        columns=["Outer Diameter",
                                 "Inter Diameter",
                                 "Donut Area"])\
                        .set_index('Outer Diameter')
print('Cross sect area of the donut circle without the hole is as follows;')
print(df_donut, '\n')

# EX13


def my_wthin_tolerance(A, a, tol):
    """This function evaluates an array to each element in a array
    at a given condition and returns the values within the given tolerance.
    my_wthin_tolerance(A, a, tol) A is the array, a is a evaluation constant,
    tol is tolerance"""
    arr_ = np.extract(abs(A-a) < tol, A)
    return arr_


c_1 = (my_wthin_tolerance(np.arange(0, 4), 1.5, 0.75))
c_2 = (my_wthin_tolerance(np.arange(0, 101, 1), 50, 3))
c_3 = np.around(my_wthin_tolerance(np.arange(0, 1.01, 0.01), 0.5, 0.03), 2)
dict_c = {'Case 1': c_1, 'Case 2': c_2, 'Case 3': c_3}
print(pd.Series(dict_c), '\n')

# EX14


def bounding_array(A, top, bottom):
    """This function evaluates an array with in given boundry conditions
    and returns the values within the given boundries.
    bounding_array(A, top, bottom) A is the array, top is a upper limit,
    botom is a lower limit"""
    arr_ = np.extract(A <= top, A)
    arr_ = np.extract(arr_ >= bottom, A)
    return arr_


arr_b = np.arange(-5, 6, 1)
top = 3
bottom = -3
a_bound = bounding_array(arr_b, top, bottom)
print(f'For a given array of {np.arange(-5, 6, 1)}.')
print(f'With boundry conditions at {top} and {bottom}.')
print(f'Yields the new array {list(a_bound)}.')
