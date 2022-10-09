#!/usr/bin/env python
# coding: utf-8
"""
Created on Tue Oct 04 20:59:35 2022

@author: rlsmi
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import fsolve
from sympy import *
import sympy as sym
# import math

print("EX1")


def my_newton(f, df, x0, tol):
    # output is an estimation of the root of f
    # using the Newton Raphson method
    # recursive implementation
    if abs(f(x0)) < tol:
        return x0
    else:
        return my_newton(f, df, x0 - f(x0)/df(x0), tol)


n = 2.6
x = 1.2
tol = 1e-8
y = Symbol('y')
f = (y**(n)) - x
fs = f
dfdy = f.diff(y)  # first order deritive)
f = lambdify(y, f)
df = lambdify(y, dfdy)

estimate = my_newton(f, df, 10, tol)
print(f" nth root = {n} and a x value = {x}\n for the function {fs}\n"
      f" derivative of {dfdy}")
print(f" root estimate = {estimate} for a tolerance of {tol}\n")

print('EX2')
print("""
There are two function such that one will equal the other
f(x) = g(x) in this case cos(x) = x^2 - 2x + 1 such that
cos(x) - (x^2 - 2x + 1) = 0. This using the bisection method
the union of the two functions can be found and the root of
the combined functions can be found. As you can see from the
graph there are two real roots and unions. Because of the
definition of the problem given we can only find one root at
a time.\n""")


def my_fixed_point(f, g, tol, max_iter):
    a = 1
    b = 2
    c = 0
    m = 0
    F = lambda m: f(m) - g(m)
    while np.abs(F(m)) > np.abs(tol) or c < max_iter:
        # finds the intersection of two function
        # where one is equal ot the other
        # by bounding x between a and b to within
        # tolerance of | f(m) | < tol with m the midpoint
        # between a and b. A check is made to limit the
        # number of iterations.

        # Create a counter
        c += 1

        # get midpoint
        m = (a + b)/2


        if np.abs(F(m)) < tol:
            # stopping condition, report m as root
            print(f'The total number of iterations {c}\n')
            return m

        elif c > max_iter:
            print(f"Exceeds maxium iterations of {max_iter}")
            m = []
            return m

        elif np.sign(F(a)) == np.sign(F(m)):
            # case where m is an improvement on a.
            a = m

        elif np.sign(F(b)) == np.sign(F(m)):
            # case where m is an improvement on b.
            b = m


f = lambda x: np.cos(x)
g = lambda x: x**2-2*x + 1
F = lambda x: f(x) - g(x)
sol = my_fixed_point(f, g, .00001, 20)
solve = optimize.fsolve(F, (-1, 2))
solve = np.round(solve, 6)
print(f'The two intersections between -1 and 2 for\n'
      f'cos(x) and x^2 - 2x + 1 are {solve[0], solve[1]}\n')
if sol == []:
    print(f"x = {sol}\n")
else:
    print(f"x = {sol}")
    print(f"cos(x) = {f(sol)}")
    print(f"x**2-2*x + 1 = {g(sol)}")
    print(f"cos(x) - (x**2-2*x + 1) = {F(sol)}")

    x = np.linspace(-.5, 2, 50)
    plt.figure(figsize=(10, 8))
    plt.plot(x, f(x), 'g--')
    plt.plot(x, g(x), 'b-')
    plt.plot(x, F(x), 'c-')
    plt.plot(sol, f(sol), c='r', marker='o', mfc='white')
    plt.plot(sol, F(sol), c='r', marker='o', mfc='white')
    plt.annotate('cos(x)', xy=(.25, 0.97),  xycoords='data',
                 xytext=(.5, 0.8), textcoords='axes fraction',
                 arrowprops=dict(arrowstyle="-|>", facecolor='black'),
                 horizontalalignment='right', verticalalignment='top',
                 )
    plt.annotate('x^2 - 2x + 1', xy=(1.75, 0.55),  xycoords='data',
                 xytext=(.8, 0.63), textcoords='axes fraction',
                 arrowprops=dict(arrowstyle="-|>", facecolor='black'),
                 horizontalalignment='right', verticalalignment='top',
                 )
    plt.annotate('cos(x) - (x^2 - 2x + 1)', xy=(-.3, -0.75),  xycoords='data',
                 xytext=(.4, 0.3), textcoords='axes fraction',
                 arrowprops=dict(arrowstyle="-|>", facecolor='black'),
                 horizontalalignment='right', verticalalignment='top',
                 )
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.show()

print('EX3')


def my_bisection(f, a, b, tol, itr, m=.1):
    """
    f = function, a and b are the bounding points,
    tol is the tolerance, m = intial guess
    """
    c = 0

    while np.abs(f(m)) > np.abs(tol) or c < itr:
        # approximates a root, R, of f bounded
        # by a and b to within tolerance
        # | f(m) | < tol with m the midpoint
        # between a and b no Recursive implementation
        # and check (c < itr) to make sure maximum iterations is not exceeded

        # Create a counter
        c += 1

        # get midpoint
        m = (a + b)/2

        if np.abs(f(m)) < tol:
            # stopping condition, report m as root
            print(f'The total number of iterations {c}')
            return m
        elif c > itr:
            print(f"Excedes maxium iterations of {itr}")
            break
        elif np.sign(f(a)) == np.sign(f(m)):
            # case where m is an improvement on a.
            a = m
        elif np.sign(f(b)) == np.sign(f(m)):
            # case where m is an improvement on b.
            b = m


# Create the function
f = lambda x: 1/x

r1 = my_bisection(f, .1, 3, 0.01, 100)
print("r1 =", r1)
r01 = my_bisection(f, .3, 3, 0.000000000001, 100)
print("r01 =", r01)

if r1 != None and r01 != None:
    print("f(r1) =", f(r1))
    print("f(r01) =", f(r01))
else:
    print("No root exist in the interval between a and b.")

print("""
The bisection method fails because it is looking for 0 in the interval |b - a|.
To be a discontinuity of the 1st kind, the function has to be defined at the
given x-value, which is not the case for 1/x at x=0. So the domain of this
function is R {0} or all reals in  the domain  [−1,0)∪(0,1].
wikipedia defines: “In mathematical analysis, the intermediate value theorem
states that if f is a continuous function whose domain contains the interval
[a, b], then it takes on any given value between f(a) and f(b) at some point
within the interval.
So for any closed interval  [a,b]⊂(−∞,0),(0,∞)  or for any closed interval
[a,b] such that it is not the case that  a ≤ 0 ≤ b  the intermediate value
theorem works.
You just have to be careful to not consider an interval that contains 0,
 since 0 is not in the domain of the function.
Note: The Intermediate Value Theorem isn’t restricted to finding where the
 function equals 0, even if it may sometimes be used to do so.\n
 """)

print('EX4')


def my_bisect(f, a, b, tol, itr=600, m=.1):
    c = 0
    R = []
    E = []

    while np.abs(f(m)) > np.abs(tol) or c < itr:
        # approximates a root, R, of f bounded
        # by a and b to within tolerance
        # | f(m) | < tol with m the midpoint
        # and check (c < itr) to make sure maximum iterations is not exceeded

        # Create a counter
        c += 1

        # get midpoint
        m = (a + b)/2

        # Create list or roots with error
        R.append(m)
        E.append(np.abs(f(m)))
        if np.abs(f(m)) < tol:
            # stopping condition, report m as root
            print(f'The total number of iterations {c}')
            return R, E
        elif c > itr:
            print(f"Excedes maxium iterations of {itr}")
            break
        elif np.sign(f(a)) == np.sign(f(m)):
            # case where m is an improvement on a.
            a = m
            # for recursive call; return my_bisection(f, m, b, tol)
        elif np.sign(f(b)) == np.sign(f(m)):
            # case where m is an improvement on b.
            b = m
            # for recursive call; return my_bisection(f, a, m, tol)
    return


f1 = lambda x: x**2 - 2  # Create the function
solve = optimize.fsolve(f1, (-2, 2))  # scipy fsolve
solve1 = np.round(solve, 6)

[R, E] = my_bisect(f1, 0, 2, 0.001)
r1 = R[-1]
print(f'The scipy fsolve solution {solve}')
print(f"R = {R}")
print(f"E = {E}\n")

f2 = lambda x: np.sin(x) - np.cos(x)
solve = optimize.fsolve(f2, (-2, 2))
solve2 = np.round(solve, 6)

[R, E] = my_bisect(f2, 0, 2, 0.001)
r2 = R[-1]
print(f'The scipy fsolve solution {solve2}')
print(f"R = {R}")
print(f"E = {E}\n")

# plot the functions and root points
x = np.linspace(-12, 12, 200)
plt.figure(figsize=(10, 8))
plt.plot(x, f2(x), 'g--', label="sin(x) - cos(x)")
plt.plot(x, f1(x), 'b-', label="x^2 - 2")
plt.plot(r1, f1(r1), c='b', marker='o', mfc='white', label='Root')
plt.plot(r2, f2(r2), c='g', marker='o', mfc='white', label='Root')
plt.ylim(-2.5, 2.5)
plt.xlim(0, 2)
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title("The roots for x^2 - 2 and sin(x) - cos(x)\n"
          "using the bisect method")
plt.grid()
plt.show()

print('EX5')


def my_newton(f, df, x0, tol):
    # Newton-Raphson Algorithm
    max_iter = 100  # Max iterations
    i = 0  # Iteration counter
    xi_1 = x0  # Initial guess
    R = []
    E = []

    # To track the initial guess
    R.append(xi_1)
    E.append(xi_1)
    # print(f'Iteration {str(i)}: x = {str(x0)} f(x) = {str(f(x0))}')

    # Iterating until either the tolerance or max iterations is met
    while np.abs(f(xi_1)) > tol or i > max_iter:
        i = i + 1
        xi = xi_1 - (f(xi_1) / df(xi_1))  # Newton-Raphson equation

        # print(f'Iteration {str(i)}: x = {str(xi)} f(x) = {str(f(xi))}')
        xi_1 = xi

        # Create the list to store root per each iteration
        R.append(xi_1)
        E.append(f(xi_1))
    return R, E


# Create the function and its derivative
x = Symbol('x')
f = x**2 - 2
f2 = sym.sin(x) - sym.cos(x)
fs = f
fs2 = f2
f_prime = f.diff(x)  # first order deritive)
f_prime2 = f2.diff(x)
f = lambdify(x, f)  # activate function
f2 = lambdify(x, f2)
df = lambdify(x, f_prime)
df2 = lambdify(x, f_prime2)

[R, E] = my_newton(f, df, 1, 1e-5)
[R2, E2] = my_newton(f2, df2, 1, 1e-5)

# Creating Data for the Line
x_plot = np.linspace(-5, 5, 50)
y_plot = f(x_plot)
y_plot2 = f2(x_plot)

# Plotting Function
fig = plt.figure(figsize=(10, 8))
plt.plot(x_plot, y_plot, 'b', label='x^2 - 2', alpha=0.6)
plt.plot(x_plot, y_plot2, 'c', label='sin(x) - cos(x)', alpha=0.6)
plt.plot(R[-1], f(R[-1]), c='b', marker='o', mfc='white', label='Root')
plt.plot(R2[-1], f2(R2[-1]), c='c', marker='o', mfc='white', label='Root')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(0, 2)
plt.ylim(-5, 5)
plt.title("The roots for x^2 - 2 and sin(x) - cos(x)\n"
          "using the Newton-Raphson method")
plt.legend()
plt.grid()
plt.show()

# Check values against scipy fsolve
roots = fsolve(f, [-2, 2])
roots2 = fsolve(f2, [-2, 2])

print(f"estimate = {R[-1]}; between 0 and 2")
print(f"{fs} = {f(R[-1])}; at x = {R[-1]}")
print(f'fsolve calculated roots:{roots}; between -2 and 2\n')
print(f"estimate = {R2[-1]}; between 0 and 2")
print(f"{fs2} = {f(R2[-1])}; at x = {R2[-1]}")
print(f'fsolve calculated roots:{roots2}; between -2 and 2\n')
print(fs)
print(f"R = {R}")
print(f"E = {E}\n")
print(fs2)
print(f"R2 = {R2}")
print(f"E2 = {E2}\n")

print('EX6')


def my_bisection(C_ocean, C_land, L, H):
    """
    f = function, a and b are the bounding points,
    tol is the tolerance, m = intial guess
    """
    itr = 120
    c = 0
    tol = 1e-6
    a = 0
    b = L
    m = (a + b)/2

    # Create the function
    # The total cost is C_ocean(√(x²+H²) + C_land*(L-x))
    # take the first derivative set it equal to 0
    # 0 = (C_ocean(x) / (√(x²+H²)) - C_land
    # solve for root of x using bisection
    f = lambda x: C_ocean*x/(np.sqrt(x**2 + H**2)) - C_land

    while np.abs(f(m)) > np.abs(tol) or c < itr:
        # approximates a root, R, of f bounded
        # by a and b to within tolerance
        # | f(m) | < tol with m the midpoint
        # between a and b no Recursive implementation
        # and check (c < itr) to make sure maximum iterations is not exceeded

        # Create a counter
        c += 1

        # get midpoint
        m = (a + b)/2

        if np.abs(f(m)) < tol:
            # stopping condition, report m as root
            # print(f'The total number of iterations {c}')  # count iterations
            print(f'The cost per mile pipeline in the ocean is {C_ocean} $/mi')
            print(f'The cost per mile pipeline in on land is {C_land} $/mi')
            print(f'The land pipe laid is {np.round(L - m, 2)} mi')
            print(f'The ocean pipe laid is {np.round(np.sqrt(m**2 + H**2), 2)}'
                  ' mi')
            print(f'The total cost is'
                  f'${np.round(C_ocean*np.sqrt(m**2+H**2) + C_land*L-m, 2)}')
            return m
        elif c > itr:
            print(f"Excedes maxium iterations of {itr}")
            break
        elif np.sign(f(a)) == np.sign(f(m)):
            # case where m is an improvement on a
            a = m
        elif np.sign(f(b)) == np.sign(f(m)):
            # case where m   is an improvement on b
            b = m


dist1 = my_bisection(20, 10, 100, 50)
print(f'The x distance that minimizes cost {dist1} mi\n')
dist2 = my_bisection(30, 10, 100, 50)
print(f'The x distance that minimizes cost {dist2} mi\n')
dist3 = my_bisection(30, 10, 100, 20)
print(f'The x distance that minimizes cost {dist3} mi\n')

print('EX7')


def my_newton(f, dfdx, x0, tol):
    # Newton-Raphson Algorithm
    max_iter = 30  # Max iterations
    i = 0  # Iteration counter
    xi_1 = x0

    print(f'Iteration {str(i)}: x = {str(x0)} f(x) = {str(f(x0))}')
    # Iterating until either the tolerance or max iterations is met
    while np.abs(f(xi_1)) > tol:
        if i > max_iter:
            print('exceeded maximum iteration')
            break
        i = i + 1
        xi = xi_1 - (f(xi_1) / dfdx(xi_1))  # Newton-Raphson equation
        print(f'Iteration {str(i)}: x = {str(xi)} f(x) = {str(f(xi))}')
        xi_1 = xi
    return xi


x = Symbol('x')
f = x**3 - 2*x + 2
fs = f
f_prime = f.diff(x) # first order deritive)
f = lambdify(x, f)
dfdx = lambdify(x, f_prime)

estimate = my_newton(f, dfdx, 1, 1e-6)
print(f'estimate {estimate}')
