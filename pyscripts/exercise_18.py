#!/usr/bin/env python
# coding: utf-8
"""
Created on Tue Sept 30 9:59:35 2022

@author: rlsmi
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sympy import *
import sympy as sym
import math


init_printing()
plt.style.use('seaborn-poster')

print('EX1')
# Create the two functions for proof of Euler Identity with sympy
x = Symbol('x')
f = sym.exp(x)
f2 = cos(x) + np.emath.sqrt(-1)*sin(x)

# Activate the two sympy function
fe = lambdify(x, f)
f2e = lambdify(x, f2)

# Set variables
ter = 20
exp = 0
x = 1
rows = []

# Iterate for summation of Taylor Series for e^ix
for i in range(ter):
    exp = exp + ((np.emath.sqrt(-1)*x)**i)/np.math.factorial(i)
    rows.append([i, exp])

# Dataframe used to store sumation of values
df_terms = pd.DataFrame(rows, columns=["terms(n)", "f(x)^(n)"])
# Format the numbers in the dataframe
pd.set_option('float_format', '{:.16f}'.format)

# Use pretty print options
pprint('Taylor Series for e^ix')
pprint(f'{i}-terms:   {exp}', use_unicode=True)
pprint(f'{f} is:  {(fe(np.emath.sqrt(-1)*x))}; for x = {np.emath.sqrt(-1)*x},')
pprint(f'            {(f2e(x))}\n         {f2} for x = {x}\n')

# Set variables
x = 1
i = np.emath.sqrt(-1)
y1 = 0
y2 = 0
ord = 12

# Taylor Series for cos(x) + I*sin(x)
for n in range(ord):
    y1 = y1 + ((-1)**n+1 * (i*x)**(2*n+1) / np.math.factorial(2*n+1))
    y2 = y2 + ((-1)**n * (x)**(2*n)) / np.math.factorial(2*n)
y = y1 + y2  # y1 = i*sin(x) + y2 = cos(x)

pprint('Taylor Series for cos(x) + i*sin(x)')
pprint(f'{ord}-terms: {y}\n', use_unicode=True)
print('\n')

print('EX2')
x = np.linspace(-np.pi/2, np.pi/2, 80)
y = np.zeros(len(x))

# Iteration series for chart label
labels = ['First Order', 'Third Order', 'Fifth Order', "Seventh Order"]

# Taylor Series for sin(x)/x
plt.figure(figsize=(10, 8))
for n, label in zip(range(4), labels):
    y = y + ((-1)**n * (x)**(2*n)) / np.math.factorial(2*n+1)
    plt.plot(x, y, label=label)

plt.plot(x, np.sin(x)/x, 'g--', alpha=0.6, label='sin(x)/x')
plt.grid()
plt.title('Taylor Series Approximations at Small x Values')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='upper center',  bbox_to_anchor=(0.25, 0.95), shadow=True)
plt.xlim(-.75, .75)
plt.ylim(0, 2.5)
plt.show()
print(f'sin(x)/x = {round(np.sin(.15)/.15, 3)} for x = 0.15')
print("""
The linear approximation L(x) = sin(a) + cos(a)*(x-a)
where a = 0; L(x) = 0 + 1*(x-0) = x, therefore x ≈ x;
for sin(x)/x ≈ x/x = 1.0 for small numbers.
This is shown above using a Taylor Series Approximations
for a small x between -0.2 to 0.2. The Taylor Series for
sin(x)/x = Σ((-1^n)(x^(2n))/(2n+1)! ;where n=0 to ∞.\n
""")

print('EX3')
x = Symbol('x')
f = sym.exp(x**2)
n = 10

# Determine the n derivatives for the function
for i in range(n):
    f = f.diff(x)
    f_new = f.diff(x)
    print(f'order {i+1}: {f}')
print('\nFor a = 0')
print('e^(x²) = x^0 + 2x^2/2! + 12x^4/4! + 120x^6/6! ...x^(2n)/n!')
print('e^(x²) = Σ(x^(2n)/n!);where n=0 to ∞.\n')


def my_double_exp(x, n):
    y = np.zeros(len(x))
    for i in range(n):
        y = y + (x)**(2*i) / np.math.factorial(i)
    return y


x = np.linspace(0, 1, 20)

ans = my_double_exp(x, n)
print(f'For the Taylor Series expansion for e^(x²) using {n} terms '
      f'from x = 0 to 1 yields;\n{ans}\n')
y = np.exp(x**2)
print(f'The computed values for e^(x²) are as follows;\n{y}')
err = np.abs(ans - y)
print()
print(f'The difference between |e^(x²)-Taylor Expansion Values| for {n}-terms'
      f' are as follows;\n{err}\n')

print('EX4')


def my_Tx_exp(n, x, a, x0):
    y = np.zeros(len(x))
    for i in range(n):
        y = y + ((x**i)/np.math.factorial(i))
    error = trunk(a, x0, n)
    return y, error


def trunk(a, x0, n):
    var('x')
    f = sym.exp(x)
    values = 20
    # Create a list of values to evaluate for c in for statement
    # to obtain the max value for function (f) evaulated a n+1 degree
    cs = [Min(x0, a) + np.abs(x0-a)*i/values for i in range(values+1)]

    fn = abs(diff(f, x, n+1))

    # Find the maximum M per cs values:
    M = Max(*[fn.subs(x, c) for c in cs])

    # Compute the error bound:
    En = (M * np.abs(x0-a)**(n+1) / np.math.factorial(n+1))
    return En


n = 7
x = np.linspace(0, .01, 10)
a = 0
x0 = max(x)
fx, err = my_Tx_exp(n, x, a, x0)

print(f'Taylor Series approximation for np.exp(x) around 0 is:{fx[1]}\n'
      f'with a maximum truncation error at {x0} of {err}\n')

print('EX5')
# Set variables
x = math.pi/2
y1 = 0
y2 = 0
y3 = 0
ord = 4

# Taylor Series for cos(x), sin(x), and cos(x)sin(x).
for n in range(ord):
    y1 = y1 + ((-1)**n * (x)**(2*n+1) / np.math.factorial(2*n+1))
    y2 = y2 + ((-1)**n * (x)**(2*n)) / np.math.factorial(2*n)
    y3 = y3 + ((-1)**n * (2*x)**(2*n+1) / np.math.factorial(2*n+1))
y3 = 0.5*y3
y4 = y1*y2
print(f'All are calculated on {ord} orders or terms')
print(f'Taylor Series for sin(x); where x=π/2 is {y1}')
print(f'Taylor Series for cos(x); where x=π/2 is {y2}')
print(f'Taylor Series for sin(x)cos(x); where x=π/2 is {y3}')
print(f'This is the multiplication of Tayor Series of sin(x) and cos(x);'
      f'where x=π/2 is {y4}')
print(f'The value for numpy sin(x)cos(x) is {np.sin(x)*np.cos(x)} ≈ 0')
print("""
The Taylor Series for sin(x)cos(x) was derived from the indentity
sin(2x) = 2sin(x)cos(x) where sin(x)cos(x) = sin(2x)/2 this yields
Taylor Series 1/2 Σ(-1)^n(2x^(2n+1)/(2n+1)!;where n=0 to ∞, for a=0.
At x=π/2 cos(x) = 1 and sin(x) = 0 if the Taylor Series estimate is
close then at this x multication of the two sin(x) and cos(x) is
more accurate, although at different x values this could change.
As a note not explained in the book ar a=0 its called a Maclaurin Series.\n
""")

print('EX6')


def truncation(a, x0, n):
    var('x')
    f = sym.exp(x)
    values = 20
    # Create a list of values to evaluate for c in for statement
    # to obtain the max value for function (f) evaulated a n+1 degree
    cs = [Min(x0, a) + np.abs(x0-a)*i/values for i in range(values+1)]

    fn = abs(diff(f, x, n+1))

    # Find the maximum M per cs values:
    M = Max(*[fn.subs(x, c) for c in cs])

    # Compute the error bound:
    En = (M * np.abs(x0-a)**(n+1) / np.math.factorial(n+1))
    return En


order = 4
x = 0.2
a = 0
cy = 0
x0 = x

# Taylor Series for cos(x).
for n in range(order+1):
    cy = cy + ((-1)**n * (x)**(2*n)) / np.math.factorial(2*n)
err = truncation(a, x0, order)
print(f'Taylor Series for  cos(x) = {cy}; where x={x0} and has order of {n}.')
print(f'Computer value for cos(x) = {math.cos(0.2)}; where x={x0}')
print(f'Has a estimated maximum truncation error at {x0} of {err}\n')

print('EX7')


def my_cosh_approximation(x, n):
    # Taylor Series for cos(x).
    coshy = np.zeros(len(x))  # needed to pass arrays
    for i in range(n):
        coshy = coshy + (x)**(2*i) / np.math.factorial(2*i)
    coshy = coshy.astype(float)
    return coshy


n = 13
x = np.linspace(0, 2*np.pi, 10)
x = x.astype(float)
y = np.zeros(len(x))

fx = my_cosh_approximation(x, n)
y = np.round(np.cosh(x), 8)
print(f'Taylor Series for  cosh(x) =\n {fx} \nwhere x=\n{x} \n'
      f'and has order of {n}.\n')
print(f'Computer values for  cosh(x) =\n {y} \nwhere x=\n{x} \n')
print(f'The difference between computer values and '
      f'Taylor Series values \n{np.abs(y - fx)}')
print("""
cosh(x) = (eⁿ + e-ⁿ)/2 where eⁿ = Σ xⁿ/n! and e-ⁿ = Σ -xⁿ/n!
therefore (eⁿ + e-ⁿ)/2 = ½ Σ 2xⁿ/2n!; simplifying Σ x²ⁿ/2n!
where n=0 to ∞, for a=0.
""")
