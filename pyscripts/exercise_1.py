"""
!/usr/bin/env python
coding: utf-8

1.6.2 Problems Python Programming and Numerical Methods
    1. Print 'I love Python' using the Python shell.
    2. Print 'I love Python' by creating a py file and running it in a shell.
    3. Type import antigravity in the iPython Shell, which will take you to
        xkcd enable you to see the awesome Python.
    4. Launch a new Juptyer notebook server in a folder called "exercise" and
        create a new Python notebook with the name "exercise_1".
        Put the rest of the problems within this notebook.
    5. Compute the area of a trangle.
    6. Comupute the surface area and volume of a cylinder.
    7. Compute the slope between two points.
    8. Compute the distance between two points.
    9. Use Pythons factorial function.
    10. Find and count all the leap years between 1500 and 2010.
    11. Use the given approximation for pi given by Ramanujan from N=0 to N=1.
"""
# Get the file location

import os
import math
from math import*
import pandas as pd
import numpy as np

cwd = os.getcwd()
print(f'The working directory is:{cwd}\n')

# Compute the area of a trangle.:


def area_tri(b, h):
    return 0.5*b*h


print('The area of a triangle with a base of 10 and height'
      ' of 12 is:', area_tri(10, 12))

# Comupute the surface area and volume of a cylinder.

vol_cly = lambda r,h: pi*(r**2)*h
surf_area = lambda r,h: 2*pi*r*h

print('The radius is 5 and height of the cylinder is 3.')
print('The volume of a cylinder is:', round(vol_cly(5, 3), 3))
print('The surface area of a cylinder is:', round(surf_area(5, 3), 3))

# Compute the slopw of a line.

slope = lambda x1, y1, x2, y2: (y2-y1)/(x2-x1)
print('The slope of a line from (3, 4) to (5, 9) is:', slope(3, 4, 5, 9))

# Compute the distance between two points.

dist_2points = lambda x1, y1, x2, y2 : sqrt((x2-x1)**2 + (y2-y1)**2)
print('The distance between two points is at (3, 4) and (5, 6) is:',
      round(dist_2points(3, 4, 5, 9), 3))

# Compute the factorial of 6.

print('The factorial of 6 is (1 x 2 x 3 x 4 x 5 x 6) is:', factorial(6), '\n')

# The number of leap years between 1500 and 2010.


def createList(r1, r2):
    return list(range(r1, r2+1))


leap = []
count = 0
non_count = 0
tot_yrs = sum(createList(1500, 2010))
start = min(createList(1500, 2010))
end = max(createList(1500, 2010))

for item in createList(1500, 2010):
    if item % 4 != 0:
        # print(item,'was not leap year.')
        non_count += 1
    elif item % 100 != 0:
        count += 1
        leap.append(item)
    elif item % 400 == 0:
        count += 1
        leap.append(item)

df = pd.DataFrame(leap, columns=['Leap Years'])
df.index = df.index+1
non_leap = non_count + 3

print(f'The total number of leap years from {start} to {end} is {count} \n')
print(f'The total number of non leap years from {start} to {end} is {non_leap}'
      ' \n')
print(df, '\n')

# approximation for pi given by Ramanujan from N=0 to N=1

"""
This section was removed due to the math function properly using
factorial(0) = 1.
def fact(x):
    print('x=', x)
    # Can not have a factorial of 0! from the factorial function in python the
    # factorial of 0 is 1.
    if x == 0:
        #return 1
    else:
        r = factorial(x)
        print('r=', r)
        return r
"""


def pi_formula():
    n = 0
    sum = 0
    i = (2*sqrt(2))/9801
    while True:
        print('n =', n)
        result = i*(factorial(4*n)/pow(factorial(n), 4))*((26390*n+1103)/pow(396, 4*n))
        print('1/pi =', result)
        sum += result  # Can not use (result += result) must be a sum of the results
        print('pi =', sum)
        print('-----------------------------------------')
        if abs(result) < 0.000000001:
            break
        n += 1
    return(1/sum)


ram_for = pi_formula()
print(f'\nPi value using Ramanujan Formula : {ram_for}')
print(f"pi = {pi}")
print('Is pi equal to Ramanujan Formula for pi:', pi == ram_for,
      '(differance =', pi - ram_for, ')\n')

# Calculate the hyperbolic sin value.

hyp_sin = lambda x: (exp(x)-exp(-x))/2
print('The equation (e^(x) - e^(-x))/2 at x=2 for the sinh =',
      round(hyp_sin(2), 12))
print("Python's hyperbolic sin function value at x=2 =", round(sinh(2), 12))
print('The above values are to a percision of 13 with a scale of 12 decimal'
      ' points.')
print('Is the equation for hyperbolic sin function equal to python sinh'
      ' fundtion:', round(hyp_sin(2), 12) == round(sinh(2), 12),'\n')

# Show that sin^2(x) + cos^2(x) at [π, π/2, π/4, π/6] = 1.0
# Numpy array was used to spped up the calculation due to vectorization.

x = np.array([pi, pi/2, pi/4, pi/6])
sin2_cos2 = lambda x: np.sin(x)**2 + np.cos(x)**2
sincos = np.around(sin2_cos2(x), decimals=15) 
# The result had to be rounded to 15 decimals to be equal
sin_1 = [1., 1., 1., 1.]

print(f'sin^2(x) + cos^2(x) at [π, π/2, π/4, π/6] = {sincos}')
print(f'An array of ones [1., 1., 1., 1.] = {sin_1}')
print('Varify that sin^2(x) + cos^2(x) at [π, π/2, π/4, π/6] is equal to one'
      ' at 15 decimal points:', sincos == sin_1)
print('Beyond 15 decimal points rounding error occure at π/4.\n')

# Calculate sin at 87°.

rads = radians(87) # Convert degrees to radians
result = sin(rads) # Calculate sin of 87° after converting 87° to radians
print('First converted 87° to radians(', round(rads, 5),')' 
      'then calculate the sin value of the converted number at',
      round(result, 5),'\n')

# Create an exception AttributeError: module 'math' has no attribute 'sni'.

try:
    math.sni()
except Exception as e: print('AttributeError:', e)

# Create an exception TypeError: math.sin() takes exactly
# one argument (0 given).

try:
    sin() 
# If the argumnet is for sin(<arg>) is left out then 
# TypeError: math.sin() takes exactly one argument (0 given)
except Exception as e: print('TypeError:', e, '\n')

# Test logical values.
L1 = (1 and (1 and not 1))
L2 = (1 and (1 and not 0))
print(L1, L2, '\n')

# Test logical values.
print((1 and 0) or (1 and not 1), '\n')

# Test logical values.
print((1 and not (1 or 1)) == (1 and not 1) and (1 and not 0), '\n')

# Test logical values.
print((1 and not (0 and not 1)), '\n')

# calculate (e^(2)*sin(pi))/6 + (ln(3)*cos(pi))/9 - 5^3
w_23 = (exp(2)*sin(pi))/6 + (log(3)*cos(pi))/9 - 5**3
print('(e^(2)*sin(pi))/6 + (ln(3)*cos(pi))/9 - 5^3 =', w_23, '\n')

# Test logical values.
a, b = 10, 25
print((a < b) and a == b, '\n')

# Test logical values.
print((1 and not 1) and (1 and not 1), '\n')
