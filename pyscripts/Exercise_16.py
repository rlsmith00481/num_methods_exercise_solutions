#!/usr/bin/env python
# coding: utf-8
"""
Created on Tue Aug 30 08:59:35 2022

@author: rlsmi
"""

import numpy as np
from scipy import optimize
# from numpy import arange
# from numpy.linalg import norm
# from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'widget')

print('EX1')
print("Repeat the multivariable calculus derivation of the least squares "
      "regression formula for an estimation function "
      "y^(x)=ax2+bx+c where a,b, and c are the parameters.")
print("""
Generalized form of the equations is y = a0 + a1x + a2x^2 + ..... + akxi^k,

The residual R^{2} ≘ ∑ from{i=1} to {n} [y_{i} - (a_{o} +  a_{1} x_{i} +
                                                  ...+ a_{k} x_{i}^k)]^2

For a polynomial of degree 2 this will lead to three eqyuations and three
unknows,
a{o}n + a_{1}∑{i=1}to{n} x{i} +...+a_{k}∑{i=1}to{n} x{i}^k = ∑{i=1}to{n} y{i}

a{o}∑{i=1}to{n} x{i} + a_{1}∑{i=1}to{n} x{i}^2 +
...+a_{k}∑{i=1}to{n} x{i}^(k+1) = ∑{i=1}to{n} x{i}y{i}

a{o}∑{i=1}to{n} x{i}^k + a_{1} ∑{i=1}to{n} x{i}^k +
...+a_{k}∑{i=1}to{n} x{i}^(2k) = ∑{i=1}to{n} x{i}^k y{i}

for 2nd order polynomial  multivariable calculus leaves 3 equations k=2:
a{o}n + a_{1}∑{i=1}to{n} x{i} + a_{k} ∑{i=1}to{n} x{i}^2 = ∑{i=1}to{n} y{i}
a{o}∑{i=1}to{n} x{i} + a{1}∑{i=1}to{n} x{i}^2 + a{2}∑{i=1}to{n} x{i}^(3) =
 ∑{i=1}to{n} x{i}y{i}
a{o}∑{i=1}to{n} x{i}^2 + a{1}∑{i=1}to{n} x{i}^3 + a{2}∑{i=1}to{n} x{i}^(4) =
 ∑{i=1}to{n} x{i}^2 y{i}
This is a 3x3 matrix X on a coefficient vector a of 3 unkowns =
 y{i}, x[i}y{i}, x{i}^2 y{i}], vector y.

[[      n,   ∑x{1}, ∑x{1}^2],
 [  ∑x{2}, ∑x{2}^2, ∑x{3}^3],
 [∑x{3}^2, ∑x{3}^3, ∑x{3}^4]]
 [a{0},
  a{1},
  a{2}] =
 [∑y{i},
  ∑y{i}x{i},
  ∑y{i}x{i}^2]. In matrix notation y = Xa, solved by XTy = XTXa,
 where XT is the transpose therefore,
  a = (XTX)^-1(XT)y""")
print("""
Note: Problem 3 is the only problem covered in the text none of the other
problems are covered.
All of the other problems will require extensive research in both linear
algebra and python regression techniques. This chapter was not well developed
or edited for mathematical accuracy.I don’t think that all of this is
necessary to fully understand the regression concepts.
It defeats the use of python and the internal module applications.
There are better books that cover this subject.
I don't see how you can actually answer the questions in this section without
additional work that goes beyond the scope of the text in this book.
To fully complete problem 1 as written would require a lengthy mathematical
proof, you have to shorthand this one.
The problems are associated with a short section at the bottm of page 279 and
top of page 280.
""")

print('EX2\n')
print("""
Write a function my_ls_params(f, x, y), where x and y are arrays of the same
size containing experimental data, and f is a list with each element a function
object to a basis vector of the estimation function. The output argument, beta,
should be an array of the parameters of the least squares regression for x, y,
and f.

The question states, "and f is a list with each element a function object to a
basis vector of the estimation function", this is very difficult to precieve in
a programming syntax, or pyhysical representation of f as a list of elemental
function objects.
I will try to present a solution to a very confusing question. It does not
request that f is to be created or if it is a function what type that is
requested is to be used.
I assume that f is y = bx + a and the solution would be in the form
beta = inv(A.T·A)·A.T·Y.
The development of the matrix was not covered in the text only infered
concerning f.
f could mean just bx as f with the formation of the matrix including a, very
confusing with out an example I will cover direct least square regression,
pseudo-inverse, lstsq method, and optimize curve fitfrom sicpy as these are
covered in the material.
I will also calculate the coefficients from the mathematical only no builtin
functions using least squares regression equations for alpha(a), beta(b).\n""")

x = np.linspace(1, 10, 30)
n = len(x)
y = np.array([20.,  24.,  27.,  32.,  36.,  42.,  48.,  54.,  63.,  70.,
              80.,  85.,  97., 105., 115., 128., 139., 147., 165., 175.,
             188., 202., 220., 227., 240., 271., 284., 305., 315., 330.])


def my_ls_parameters(x, y):

    print('Direct least square regression')
    x_points = x[18:30]
    y_points = y[18:30]

    # create the matirx
    M = np.vstack([x_points, np.ones(len(x_points))]).T

    # make y a column vector
    y_plot = y_points[:, np.newaxis]
    beta1 = np.dot((np.dot(np.linalg.inv(np.dot(M.T, M)), M.T)), y_plot)

    print('beta vector for a1 and a2 in y = a1(x) + a2 \n '
          'for x = 7.2 to 10.0\n', beta1, '\n')

    print('Pseudo-inverse')
    x_pinv = x[0:10]
    y_pinv = y[0:10]
    M_pinv = np.vstack([x_pinv, np.ones(len(x_pinv))]).T
    pinv = np.linalg.pinv(M_pinv)
    beta2 = pinv.dot(y_pinv)
    print('beta vector for a1 and a2 in y = a1(x) + a2 \n'
          ' for x = 1.0 to 3.7\n', beta2, '\n')

    print('lstsq method')
    x_sq = x[10:18]
    y_sq = y[10:18]
    M_sq = np.vstack([x_sq, np.ones(len(x_sq))]).T
    beta3 = np.linalg.lstsq(M_sq, y_sq, rcond=None)[0]
    print('beta vector for a1 and a2 in y = a1(x) + a2 \n'
          ' for x = 3.8 to 6.1\n', beta3, '\n')
    return beta1, beta2, beta3


alpha1, alpha2, alpha3 = my_ls_parameters(x, y)

print('Optimize curve fit from sicpy')


def funct(x, a, b, c):
    y = a*x**2 + b*x + c
    return y


popt, pcov = optimize.curve_fit(funct, x, y)  # Opt curve fit
model = LinearRegression()  # create a model
X = np.vstack([x, np.ones(len(x))]).T
model.fit(X, y)  # fit the model
r_squared = model.score(X, y)  # Calculate the R^2 fit ot data
perr = np.sqrt(np.diag(pcov))  # Calculate the deg of error in coef

# calculating regression coefficients
a, b, c = popt

print('r_squared:', r_squared)
print(f'The estimated function is: y = {a: .1f}*x^2 + {b: .1f}*x + {c: .1f}')
print(f'The error: {perr}')

# define a sequence of inputs between the smallest and largest known inputs
x_line = x  # could use arange(min(x), max(x), 1) to specify a range of data

# calculate the output for the range
y_line = funct(x_line, a, b, c)

# round the results from x data for chart
x1 = np.round(x[18:30], 2)
x2 = np.round(x[0:10], 2)
x3 = np.round(x[10:18], 2)

plt.figure(figsize=(8, 6))
plt.plot(x, y, 'bo', label="Data", alpha=0.3)
plt.plot(x, (alpha1[0]*x + alpha1[1]), 'k',
         label=f'Direct least square regression for x ={x1}')
plt.plot(x, (alpha2[0]*x + alpha2[1]), 'g',
         label=f'Pseudo-inverse for x = {x2}')
plt.plot(x, (alpha3[0]*x + alpha3[1]), 'r',
         label=f'Using Numpy lstsq method for x = {x3}')
plt.plot(x_line, y_line, color='gold',
         label=f'Optimize curve fit from sicpy using '
         f'f(x) = {a: .2f}*x^2 + {b: .2f}*x + {c: .2f}')
plt.legend(fontsize=7)
plt.xlabel('x')
plt.ylabel('y')
plt.ylim(0, 400)
plt.grid()
plt.show()

x = np.linspace(1, 10, 30)
n = len(x)
y = np.array([20.,  24.,  27.,  32.,  36.,  42.,  48.,  54.,  63.,  70.,
              80.,  85.,  97., 105., 115., 128., 139., 147., 165., 175.,
             188., 202., 220., 227., 240., 271., 284., 305., 315., 330.])


def my_linear_coef(x, y):
    n = len(x)  # Total points

    # A check of all the sums of the data to get the linear equation.
    # f_linear = [n, np.sum(y), np.sum(x), np.sum(x**2), np.sum(x*y)]

    # calculating regression coefficients
    b_0 = (np.sum(y)*np.sum(x**2) - np.sum(x)*np.sum(x*y))/ \
          ((n*np.sum(x**2) - np.sum(x)**2))
    b_1 = (n*np.sum(x*y) - np.sum(x)*np.sum(y))/ \
          (n*np.sum(x**2) - np.sum(x)**2)
    beta = [b_0, b_1]
    return beta


# A different way to calculate the coef of the equations in vector format.
# Same as above
def estimate_coef(x, y):
    # number of observations/points
    n = np.size(x)

    # mean of x and y vector
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # calculating cross-deviation and deviation about x
    xy_dev = np.sum(y*x) - n*x_mean*y_mean
    x_dev = np.sum(x**2) - n*x_mean**2

    # calculating regression coefficients
    b_1 = xy_dev / x_dev
    b_0 = y_mean - b_1*x_mean
    B = [b_0, b_1]
    return B


def my_poly_coef(x, y):
    # setting up the parameters
    n = len(x)
    xi = np.sum(x)
    xi2 = np.sum(x**2)
    xi3 = np.sum(x**3)
    xi4 = np.sum(x**4)
    yi = np.sum(y)
    xiyi = np.sum(x*y)
    xi2yi = np.sum(x**2*y)

    # polynomial fit matrix
    X = np.array([[n, xi, xi2], [xi, xi2, xi3], [xi2, xi3, xi4]])
    y_vec = np.array([yi, xiyi, xi2yi])

    # calculating regression coefficients for second order poly
    a = np.dot((np.dot(np.linalg.inv(np.dot(X.T, X)), X.T)), y_vec)
    return a


def my_ls_params(f, x, y):
    y_pred = f[0] + f[1]*x
    return y_pred


def my_ls_params_poly2(a, x, y):
    y_poly = a[0] + a[1]*x + a[2]*x**2
    return y_poly


# Using the coef vector over a limit in the data
alpha = my_poly_coef(x, y)
beta2 = my_linear_coef(x[18:30], y[18:30])
beta3 = my_linear_coef(x[0:10], y[0:10])
beta4 = my_linear_coef(x[10:18], y[10:18])

# getting the y value of the plot
Y1 = my_ls_params_poly2(alpha, x, y)
Y2 = my_ls_params(beta2, x, y)
Y3 = my_ls_params(beta3, x, y)
Y4 = my_ls_params(beta4, x, y)

# plotting the data
plt.figure(figsize=(8, 6))
plt.plot(x, y, 'bo', label="Data", alpha=0.3)

plt.plot(x, Y1, color='gold',
         label=f'Poly second order least square regression for '
         f'f(x) = {alpha[0]: .2f}*x^2 + {alpha[1]: .2f}*x + {alpha[2]: .2f}')

# for plot labels
x1 = np.round(x[18:30], 2)
x2 = np.round(x[0:10], 2)
x3 = np.round(x[10:18], 2)

plt.plot(x, Y2, 'k', label=f'least square regression for x values ={x1}')
plt.plot(x, Y3, 'g', label=f'least square regression for x values ={x2}')
plt.plot(x, Y4, 'r', label=f'least square regression for x values ={x3}')

plt.legend(fontsize=7)
plt.xlabel('x')
plt.ylabel('y')
plt.ylim(0, 400)
plt.grid()
plt.show()

print('EX3')
x = np.linspace(1, 30, 50)
y = np.array([.02,  .08,   .21,    .43,
              .81, 1.25,   2.2,    2.8,
              3.8,  5.2,   6.8,    8.3,
              11.6, 13.2,  15.5,   18.3,
              25.2, 27.1,  34.2,   38.6,
              44.3, 48.4,  58.1,   62.3,
              66.9, 74.3,  88.1,   97.9,
             110.5, 120.8, 134.2, 140.3,
             164.5, 173.1, 192.7, 204.8,
             222.1, 245.5, 254.2, 281.3,
             323.4, 342.5, 355.7, 370.6,
             395.4, 402.9, 449.7, 462.5,
             508.8, 548.1])


def my_function_fit(x, y):
    n = len(x)
    print(f"The total number of data points: {n}")

    # calculating regression coefficients for linear regression power law
    b = (n*(np.sum(np.log(x)*np.log(y))) - np.sum(np.log(x))
         *np.sum(np.log(y)))/(n*np.sum(np.log(x)**2) - (np.sum(np.log(x)))**2)
    a = (np.sum(np.log(y)) - b*(np.sum(np.log(x))))/n

    # converting alpha back from log values
    a = np.exp(a)
    print(f"The coefficients for {a: .2f}x^{b: .2F} where: ax^b")
    return a, b


alpha, beta = my_function_fit(x, y)

# Calculate y for the using alpha and beta
Y1 = alpha*x**beta

# plotting the data
plt.figure(figsize=(8, 6))
plt.plot(x, y, 'bo', label="Data", alpha=0.3)
plt.plot(x, Y1, 'k', label=f"Least squares power function for f(x) = "
         f"{alpha: 0.2f}x^{beta: 0.2f}")
plt.legend(fontsize=7, loc='upper left')
plt.xlabel('x')
plt.ylabel('y')
plt.ylim(0, 600)
plt.grid()
plt.show()

print('EX4')

xi = np.array([5, 7.5, 12.5, 17.5])
yi = np.array([4, 10.5, 33.5, 82])
x1 = np.array([5, 7.5, 12.5, 17.5, 18.5])
y1 = np.array([4, 10.5, 33.5, 82, 96])
x2 = np.linspace(1, 30, 30)
y2 = np.array([.02,  .08,   .21,    .43,
               .81, 1.25,   2.2,    2.8,
               3.8,  5.2,   6.8,    8.3,
               11.6, 13.2,  15.5,   18.3,
               25.2, 27.1,  34.2,   38.6,
               44.3, 48.4,  58.1,   62.3,
               66.9, 74.3,  88.1,   97.9,
              110.5, 120.8])


def finding_error(x, y):
    # finding regression coefficients for linear regression cubic with
    # np.polyfit
    a = np.polyfit(x, y, 3)
    n = len(x)

    # Collecting predicted y values
    error = []
    pred_y = a[3] + a[2]*x + a[1]*x**2 + a[0]*x**3
    for i in range(len(x)):
        ei = np.abs(y[i] - pred_y[i])
        error.append(ei)
    erv = np.sum((y-pred_y)**2)
    ern = np.linalg.norm(error)**2
    print(f"Total squared error for {n} data points is:{erv: .4f}")
    print(f'Norm of error for {n} data points: {ern: .4f}')
    return pred_y, a


y0, a = finding_error(xi, yi)
y1, a1 = finding_error(x1, y1)
y2, a2 = finding_error(x2, y2)
print("""
Can we place another data point (x, y) such that no additional error
incurred for the estimation function?

No, each addition point will increase the total sum of the area between
the estimation function and the actual data no matter how small each is.
This is shown in the data and plot below.
""")
plt.figure(figsize=(8, 6))
plt.plot(xi, yi, 'bo', label="Data 4 points", alpha=0.3)
plt.plot(x1, y1, 'go', label="Data2 5 points", alpha=0.3)
plt.plot(x2, y2, 'ro', label="Data3 30 points", alpha=0.3)
plt.plot(xi, y0, 'k', label=f"Cubic polynomial LSR f(x) 4 points ="
         f" {a[0]: .2f}x^3+{a[1]: .2f}x^2+{a[2]: .2f}x+{a[3]: .2f}")
plt.plot(x1, y1, 'g', label=f"Cubic polynomial LSR f(x) 5 points ="
         f" {a1[0]: .2f}x^3+{a1[1]: .2f}x^2+{a1[2]: .2f}x+{a1[3]: .2f}")
plt.plot(x2, y2, 'r', label=f"Cubic polynomial LSR f(x) 30 points ="
         f" {a2[0]: .2f}x^3+{a2[1]: .2f}x^2+{a2[2]: .2f}x+{a2[3]: .2f}")
plt.legend(fontsize=7, loc='upper left')
plt.xlabel('x')
plt.ylabel('y')
plt.ylim(0, 150)
plt.grid()
plt.show()

print('EX5')

x = np.linspace(0, 2*np.pi, 1000)
y = 3*np.sin(x) - 2*np.cos(x) + np.random.random(len(x))
f = [np.sin, np.cos]


def my_lin_regression(f, x, y):
    # Develop the matrix [1, cos(x), sin(x)] for n number of entries
    # where n = number of x values
    a0 = np.ones(len(x))
    a1 = f[1](x)
    a2 = f[0](x)

    # Create the matrix from the arrays a0=1, a1=cos(x), a2=sin(x)
    # for n number of entries where n = number of x values.
    A = np.vstack([a0, a1, a2]).T

    # Least squares regression formula
    b = np.dot((np.dot(np.linalg.inv(np.dot(A.T, A)), A.T)), y)

    # Reverse the array order of numbers to match the given plot data
    b = b[::-1]
    return b


def r2(x, y):
    sxx = np.sum((x-np.mean(x))**2)
    syy = np.sum((y-np.mean(y))**2)
    sxy = np.sum(x*y) - len(x)*np.mean(x)*np.mean(y)
    r2 = sxy**2/(sxx*syy)
    return r2


beta = my_lin_regression(f, x, y)
residuals = y - beta[0]*f[0](x)+beta[1]*f[1](x)+beta[2]
err = np.linalg.norm(residuals)**2
R2 = r2(x, (beta[0])*f[0](x)+(beta[1])*f[1](x)+(beta[2]))

plt.figure(figsize=(10, 8))
plt.plot(x, y, 'b.', label='data')
plt.plot(x, beta[0]*f[0](x)+beta[1]*f[1](x)+beta[2], 'r',
         label=f'y = {beta[0]: .3f}sin(x) + {beta[1]: .3f}cos(x) + '
         f'{beta[2]: .3f}, for {len(x)} data points.\nR2 = {R2: .5f}')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Least Square Regression Example\n')
plt.legend()
plt.show()

x = np.linspace(0, 1, 1000)
y = 2*np.exp(-0.5*x) + 0.25*np.random.random(len(x))


def my_exp_regression(x, y):
    a0 = np.sum(y)
    a1 = np.sum(x*y)
    a2 = np.sum(x*y)
    a3 = np.sum(x**2*y)
    y0 = np.sum(y*np.log(y))
    y1 = np.sum(x*y*np.log(y))
    A = np.array([[a0, a1], [a2, a3]])
    Y = np.array([y0, y1])
    b = np.dot((np.dot(np.linalg.inv(np.dot(A.T, A)), A.T)), Y)

    # Convert back b[0] as defined b[0] = exp(a) and b[1] = b[1]
    b[0] = np.exp(b[0])
    return b[0], b[1]


# Function for correlation coefficient
def r2(x, y):
    sxx = np.sum((x-np.mean(x))**2)
    syy = np.sum((y-np.mean(y))**2)
    sxy = np.sum(x*y) - len(x)*np.mean(x)*np.mean(y)
    r2 = sxy**2/(sxx*syy)
    return r2


# Obtain alpha and beta values to use in LSRF equation.
alpha, beta = my_exp_regression(x, y)

# Calculate correlation coefficient.
R2 = r2(x, alpha*np.exp(beta*x))

plt.figure(figsize=(10, 8))
plt.plot(x, y, 'b.', label='data')
plt.plot(x, alpha*np.exp(beta*x), 'r',
         label=f'y = {alpha: .3f}e^({beta: .3f}x),\
 for {len(x)} data points.\n R2 = {R2: .5f}')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Least Square Regression on Exponential Model')
plt.legend()
plt.show()
