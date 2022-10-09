#!/usr/bin/env python
# coding: utf-8
"""
Created on Mon Sept 12 09:41:10 2022

@author: rlsmi
"""
import numpy as np
# import sympy as sym
import time
from numpy.linalg import inv, qr

print('EX1, EX2, EX3')
to = time.time()

print("""Write down the characteristic equation for matrix """)
A = np.array([[3, 2], [5, 3]])
print('A = ')
print(A)
print("""
A = [[(3-ʌ), 2],
     [5, (3-ʌ)]]\n
Characteristic Equation
determinate = (3-ʌ)(3-ʌ)-10 = (ʌ^2 - 6ʌ + 9) - 10 = ʌ^2 - 6ʌ - 1\n""")
print('Using numpy linear algebra functions for eigen values/vectors:')


def eigenvalues_vectors(A):
    for i in range(len(A)-1):
        w, v = np.linalg.eig(A)
        eigenvalues = {1: w[i], 2: w[i+1]}
        eigenvectors = {1: v[:, i], 2: v[:, i+1]}
        print("Is Ax = ʌx:")
        print(f'For {w[i]}: ', np.allclose(np.dot(A, v[:, i]),
                                           np.dot(w[i], v[:, i])))
        print(f'For {w[i+1]}: ', np.allclose(np.dot(A, v[:, i+1]),
                                             np.dot(w[i+1], v[:, i+1])))
    return eigenvalues, eigenvectors


eigenvalues, eigenvectors = eigenvalues_vectors(A)
print(f'The eigenvalues: {eigenvalues}')
print(f'The eigenvectors: {eigenvectors}\n')
ti = time.time()
print(f'Delta Time: {ti-to} sec')
print('____________________________________________________________________\n')
print('Without numpy linear algebra functions for eigen values/vectors:\n')
t0 = time.time()


def roots(a, b, c):
    x1 = (-b + np.sqrt(b**2 - 4*a*c))/2*a
    x2 = (-b - np.sqrt(b**2 - 4*a*c))/2*a
    return x1, x2


def two_norm(x):
    two_norm = 0
    for i in x:
        two_norm += np.abs(i)**2
    two_norm = np.sqrt(two_norm)
    return two_norm


def rref(mat, precision=0, GJ=False):
    m, n = mat.shape
    p, t = precision, 1e-1**precision
    A = np.around(mat.astype(float).copy(), decimals=p)
    if GJ:
        A = np.hstack((A, np.identity(n)))
    pcol = -1  # pivot colum
    for i in range(m):
        pcol += 1
        if pcol >= n: break
        # pivot index
        pid = np.argmax(abs(A[i:, pcol]))
        # Row exchange
        A[i, :], A[pid+i, :] = A[pid+i, :].copy(), A[i, :].copy()
        # pivot with given precision
        while pcol < n and abs(A[i, pcol]) < t:
            # pivot index
            pid = np.argmax(abs(A[i:, pcol]))
            # Row exchange
            A[i, :], A[pid+i, :] = A[pid+i, :].copy(), A[i, :].copy()
            pcol += 1
        if pcol >= n: break
        pivot = float(A[i, pcol])
        for j in range(m):
            if j == i: continue
            mul = float(A[j, pcol])/pivot
            A[j, :] = np.around(A[j, :] - A[i, :]*mul, decimals=p)
        A[i, :] /= pivot
        A[i, :] = np.around(A[i, :], decimals=p)

    if GJ:
        return A[:, :n].copy(), A[:, n:].copy()
    else:
        return A


x1, x2 = roots(1, -6, -1)
eigenvals = {1: x1, 2: x2}
print(f'The eigenvalues: {eigenvals}\n')
B = np.array([[3-x1, 2], [5, 3-x1]])
B2 = np.array([[3-x2, 2], [5, 3-x2]])
Br = rref(B, precision=10, GJ=False)
Br2 = rref(B2, precision=10, GJ=False)

print(f'Reduced echelon matrix from the eigen value {x1}')
print(Br2, '\n')
vec = [Br2[0][1], Br2[0][0]]
print(f'Basis eigen vector for {x1}')
print({1: vec[0], 2: vec[1]}, '\n')
vecn = 1/two_norm(vec)
print(f"""
1/normalized basis eigen vector {vecn} multiplied by eigen basis vector
{vec} to get eigen vector for the eigen value {x1}.\n""")
vec = np.dot(vecn, vec)
print(f'The eigen vector for {x1}')
print({1: vec[0], 2: vec[1]}, '\n')
print(f'For Ax = ʌx at eigen value of {x1}: ',
      np.allclose(np.dot(A, vec), np.dot(x1, vec)))
print(f'Ax = {np.dot(A,vec)}')
print(f'ʌx = {np.dot(x1,vec)}', '\n')

print(f'Reduced echelon matrix from the eigen value {x2}')
print(Br, '\n')
vec2 = [Br[0][1], Br[0][0]]
print(f'Basis eigen vector for {x2}')
print({1: vec2[0], 2: vec2[1]}, '\n')
vecn2 = 1/two_norm(vec2)
print(f"""
1/normalized basis eigen vector {vecn2} multiplied by eigen basis vector
{vec2} to get eigen vector for the eigen value {x2}.\n""")
vec2 = np.dot(vecn2, vec2)
print(f'The eigen vector for {x2}')
print({1: vec2[0], 2: vec2[1]}, '\n')
print(f'For Ax = ʌx at eigen value of {x2}: ',
      np.allclose(np.dot(A, vec2), np.dot(x2, vec2)))
print(f'Ax = {np.dot(A,vec2)}')
print(f'ʌx = {np.dot(x2,vec2)}\n')
t1 = time.time()
print(f'delta time: {t1-t0}sec')
print(f"Run time difference between the two methods:{(t1-t0)-(ti-to)} sec")

print('EX4, EX5, EX6, EX7')
print('Power Method')


def normalize(x):
    fac = abs(x).max()
    x_n = x / x.max()
    return fac, x_n


def two_norm(x):
    two_norm = 0
    for i in x:
        two_norm += np.abs(i)**2
    two_norm = np.sqrt(two_norm)
    return two_norm


x = np.array([1, 1, 1])
a = np.array([[2, 1, 2],
              [1, 3, 2],
              [2, 4, 1]])
ar = rref(a, precision=10, GJ=True)
print('Reduced matrix and inital matrix\n', inv(ar))
tol = 0.00001
count, xn = 0, 0
diff = 1
# for i in range(8):
while diff > tol:
    _, xn = normalize(x)
    x = np.dot(a, x)
    lambda_1, x = normalize(x)
    diff = np.abs(np.sum(xn) - np.sum(x))
    count += 1
xn = two_norm(x)
xn = -1/xn

print(f'Eigenvalue: {np.round(lambda_1, 5)}')
print(f'Eigenvector: {np.round(np.dot(xn,x), 5)}')
print(f'Number of iterations: {count}')
print('\n ______________________________\n')

print('Inverse Power Method')
tol = 0.00001
a_inv = inv(a)

count, xn = 0.0, 0.0
diff = 1
# for i in range(1000):
while diff > tol:
    _, xn = normalize(x)  # for tolerance
    x = np.dot(x, a_inv)
    lambda_1, x = normalize(x)
    diff = np.abs(np.sum(x) - np.sum(xn))
    count += 1

print(f'Eigenvalue: {np.round(lambda_1, 8)}')
print(f'Eigenvector: {np.round(x, 8)}')
print(f'Number of iterations: {count}')

print('\n ______________________________\n')

q, r = qr(a)
print('Q:')
print(q, '\n')
idq = np.dot(inv(q), q)
print('Show below the inv(Q) is equal Q.T.')
print('The inverse of Q:')
print(np.round(inv(q), 5))
print('The transpose of Q:')
print(np.round(q.T, 5))
print('The transpose of Q = inverse of Q the matrix is orthogonal:')
print(np.round(inv(q), 5) == np.round(q.T, 5), '\n')

print('The inverse of Q matrix * Q is equal to identity matrix.')
print("Q'Q = I the matrix is orthogonal")
print(idq)
print('R:')
print(r, '\n')

b = np.dot(q, r)
print('QR:')
print(b, '\n')

p = [100, 250, 500, 750, 1000, 1250]
for i in range(1250):
    q, r = qr(a)
    a = np.dot(r, q)
    if i+1 in p:
        print(f'Iteration {i+1}:')
        print(a, '\n')
w, v = np.linalg.eig(a)
print(w, '\n')

print('EX8')


def eigenvalues_vectors(A):
    w, v = np.linalg.eig(A)
    num = [i+1 for i in range(len(A))]
    eigenvalues = dict(zip(num, w))
    eigenvectors = dict(zip(num, v))
    print("Is Ax = ʌx:")
    for i in range(len(A)):
        print(f'For {w[i]}: ', np.allclose(np.dot(A, v[:, i]),
                                           np.dot(w[i], v[:, i])))
    return eigenvalues, eigenvectors


x = np.array([1, 1, 1])
a = np.array([[2, 1, 2],
              [1, 3, 2],
              [2, 4, 1]])

eigenvalues, eigenvectors = eigenvalues_vectors(a)
print('The eigenvalues:')
for key, value in eigenvalues.items():
    print(key, ' : ', value)
print('The eigenvectors:')
for key, value in eigenvectors.items():
    print(key, ' : ', value)
