#!/usr/bin/env python
# coding: utf-8
"""
Created on Mon Sept 12 11:29:08 2022

@author: rlsmi
"""
import numpy as np
from numpy.linalg import norm
from scipy.linalg import lu
import pymatrix
import sympy
import sys
from collections import defaultdict
import string

print('EX1')
print("""
It is strongly recomended that you read a book on linear
algebra, which will give you greater mastery of the contents
of this chapter. We strongly recommend reading the first
part of the book Optimization Models by Giuseppe Calcfiore
and Laurent El Ghaoui to get you started.\n""")

print('EX2')
print("""
Show that matrix multiplication distributes over matrix
addition: show A(B+C)=AB+AC assuming that A,B, and C
are matrices of compatible size.\n""")
A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
B = [[10, 11, 12], [13, 14, 15], [16, 17, 18]]
C = [[1, 5, 8], [4, 9, 5], [6, 3, 9]]
print('A(B+C):')
print(np.matmul(A, (np.add(B, C))), '\n')
print('AB+AC:')
print(np.matmul(A, B) + np.matmul(A, C), '\n')
print(f'Is it true that A(B+C)=AB+AC\n'
      f'{(np.matmul(A, (np.add(B, C)))) == (np.matmul(A,B) + np.matmul(A,C))}\n')

print('EX3')


def my_orthogonal(v1, v2, tol):
    v1 = v1.T
    # Test to see if the angle between the vectors is orthogonal
    th = np.arccos(np.dot(v1, v2)/(norm(v1)*norm(v2)))
    if (np.pi/2 - th) < tol:
        c1 = 1
        return c1
    else:
        c2 = 0
        return c2


print('case1')
u1 = np.array([[1], [0.001]])
print('u1:\n', u1)
v1 = np.array([[0.001], [1]])
print('v1:\n', v1)
tol1 = 0.01
print('Tolerance:', tol1)
case1 = my_orthogonal(u1, v1, tol1)
print(f'For the first case we have: {case1}\n')

print('case2')
u2 = np.array([[1], [0.001]])
print('u2:\n', u2)
v2 = np.array([[0.001], [1]])
print('v2:\n', v1)
tol2 = 0.001
print('Tolerance:', tol2)
case2 = my_orthogonal(u2, v2, tol2)
print(f'For the second case we have: {case2}\n')

print('case3')
u3 = np.array([[1], [0.001]])
print('u3:\n', u3)
v3 = np.array([[1], [1]])
print('v3:\n', v3)
tol3 = 0.01
print('Tolerance:', tol3)
case3 = my_orthogonal(u3, v3, tol3)
print(f'For the third case we have: {case3}\n')

print('case4')
u4 = np.array([[1], [1]])
print('u4:\n', u4)
v4 = np.array([[-1], [1]])
print('v4:\n', v4)
tol4 = 0.0000000001
print('Tolerance:', tol3)
case4 = my_orthogonal(u4, v4, tol4)
print(f'For the fourth case we have: {case4}\n')

print('EX3')

s1 = """
Zelda Rae Williams (born July 31, 1989)[1][2] is an American actress,
director, producer, and writer. She is the daughter of the late actor
and comedian Robin Williams, and film producer and philanthropist
Marsha Garces Williams."""

s2 = """
Linear algebra is the foundation of many engineering fields.
Vectors can be considered as points in Rn; addition and multiplication
are defined on them, although not necessarily the same as for scalars.
A set of vectors is linearly independent if none of the vectors can be
written as a linear combination of the others. Matrices are tables of
numbers. They have several important properties including the determinant,
rank, and inverse. A system of linear equations can be represented by the
matrix equation Ax=y."""

tol = (np.pi/4)/2


def my_is_simular(s1, s2, tol):
    # create two arrays v1 and v1 fill with 25 '0'.
    v1 = np.array([0]*25)
    v2 = np.array([0]*25)
    # create a dictionary to hold charaters.
    # default_factory: A function returning the default value for the
    # dictionary defined. If this argument is absent then the dictionary
    # raises a KeyError.
    chars1 = defaultdict(int)
    chars2 = defaultdict(int)
    # Create a list of lower case alphabetical ordered charaters a - z
    alph = list(string.ascii_lowercase)
    # create a dictionary of th count alphabetical ordered charaters
    # from s1 and s2
    for char in s1:
        chars1[char] += 1

    for char in s2:
        chars2[char] += 1
    # assign values to vectors based on charater counts in s1 and s2.
    for i in range(25):
        v1[i] = chars1[alph[i]]
        v2[i] = chars2[alph[i]]
    v1 = v1.T
    # find the angle between the vectors v1 and v2
    th = np.arccos(np.dot(v1, v2)/(norm(v1)*norm(v2)))
    print(f"The angle betwen the two vectors is {round(th*180/np.pi, 3)}°")
    print(f"Tolerance is at {round(tol*180/np.pi, 3)}°")
    # test to see if the angle is below or above a tolerance.
    # What plane or quadant.
    if np.abs(th) < tol:
        c1 = 1
        return c1
    else:
        c2 = 0
        return c2


print('The two paragraphs used to assign values to a vector from a-z:')
print(s1)
print(s2, '\n')
code = my_is_simular(s1, s2, tol)
print(f'Flag is 1 for values < the {tol*180/np.pi} and 0 values > '
      f'{tol*180/np.pi}')
print(f'The flag is: {code}')

print('EX4')

M = np.array([[12, 24, 0, 11, -24, 18,  15],
              [19, 38, 0, 10, -31, 25,   9],
              [ 1,  2, 0, 21,  -5,  3,  20],
              [ 6, 12, 0, 13, -10,  8,   5],
              [22, 44, 0,  2, -12,  17, 23]])


def my_make_lin_ind(A):
    print("""
    With the help of sympy.Matrix().rref() method, we can put a matrix into
    reduced Row echelon form. Matrix().rref() returns a tuple of two elements.
    The first is the reduced row echelon form, and the second is a tuple of
    indices of the pivot columns.\n""")
    # to get rows you need the transpose of A matrix
    _, inds = sympy.Matrix(A).rref()
    indp = list(inds)  # create a list from the tuple
    rows, columns = A.shape
    # Create the empty matrix base B on rows of the A matrix
    # and columns of linearly independent columns
    B = np.empty(shape=(rows, len(indp)), dtype='object')
    c = 0  # Counter 0 to number of v counts
    # Fill the empty matrix
    for v in indp:
        B[:, c] = A[:, v]
        c += 1
    return B, _, indp


B, reduced, indp = my_make_lin_ind(M)
reduced = np.array(reduced)
print(f'The reduced matrix is:\n {reduced}\n\n '
      f'The linearly independent columns are:\n {indp}\n')
print(f'Test case for found linearly independent columns of matrix:\n {B}\n')

myMatrix = pymatrix.matrix([[1,  2, -4,  8],
                           [3,  2,  3,  4],
                           [2,  0,  3, -2],
                           [1, -7,  6,  5]])

print("The matrix is:")
print(myMatrix, '\n')
# Computing the determinant of the matrix
d = myMatrix.det()
print(f"The determinant of the given matrix is: {d}\n")

print('EX5')


def my_rec_det(arr):
    n = len(arr)
    if n == 1: return arr[0][0]
    if n == 2: return arr[0][0]*arr[1][1] - arr[0][1]*arr[1][0]
    sum = 0

    def minor(m, i, j):
        return [row[:j] + row[j+1:] for row in (m[:i]+m[i+1:])]

# (i,j)th minor of a matrix of size n is a smaller matrix of size n-1
# with the i'th row and j'th column deleted. [row for row in (m[:i] + m[i+1:])]
# m[:i] gives the first i rows of m (remember that we are representing
# a matrix as a list of rows, and adding two lists gives back a bigger list
# in python),
# row[:j] + row[j+1:] This gives us all the elements of a row except the j'th
# element (lst[a:b] gives a list with elements from lst between a and b, with
# b excluded) Combine the above two and you get an expression which returns
# a new matrix with i'th row and j'th column excluded:minor matrix.

    for i in range(0, n):
        m = minor(arr, 0, i)
        sum = sum + ((-1)**i)*arr[0][i] * my_rec_det(m)
        # This is the jth column expansion by cofactors
        # |A| = ∑i=1 to n aijCij = a1jC1J + a2jC2j+..anjCnj
    return sum


M = [[1,  2, -4,  8],
     [3,  2., 3,  4],
     [2,  0,  3, -2],
     [1, -7,  6,  5]]
print("The matrix is:")
print(np.stack(M), '\n')
d = float(my_rec_det(M))
print(f"The determinant of the given matrix is: {'%.1f' % d}")
print("""
A correction needs to made here, the calculation of the determinate
is not the Cramer's rule. Cramer's rule uses the determinate to find
a solution to a set of linear equation that has nxn size. The determinate
is still calculated based on system of roll elimination using cofactors
which are calculated from minors.\n""")

print('EX6')
print("""
What is the complexity of my_rec_det in the previous problem?
Do you think this is an effective way of determining if a
matrix is singular or not?""")
print("""
There is some that think it is not reasonable to assume
constant-time arithmetic without more information.
In fact the O(n3) estimate is only realistic over finite fields.
Also, there seems to be an algorithm for computing the determinant
based on the minor expansion which uses dynamic programming
techniques and has a running time of O(n4).

No, it is not an effective way to determine if a
matrix is singular or not. A better way is to find
the determinate if 0 then singular.""")

print('EX7')
print("""
Some corrections need to made with respect to the nomenclatures
used in the question. The p vector is the R3 basis vector derived
from the polynomial The onginial polynomial is considered P2 and
the derivative polynomial is P.The D matrix is generally considered
to be the transform matrix T or A and the resulting vector is the
R2 basis vector which yields the derivative of P2 called P. The
basis vector used to derive the T transform matrix for a second
order polynomial is '{1, x, x^2}' Such as T(1) = 0(1) + 0(x)
T(x) = 1(1) + 0(x), T(x^2) = 0(1) + 2(x).
input order for R3 or p c(2)x^2 + c(1)x + c(0) [c(2), c(1), c]
""")


def my_ploy_der_mat(p):
    r3 = np.array(p)
    r3 = r3[::-1]  # switch the order for multipcation and readabiltiy
    n = len(r3)  # length of basis vector R3
    m = n-1  # used to form the T transformation matrix
    # Create a matrix and fill it
    D = [0] * m
    for x in range(m):
        D[x] = [0] * n
        D[x][x+1] = x+1
    r2 = np.dot(D, r3)  # Obtain the R2 basis vector
    r2 = r2[::-1]  # switch to match the shape of polynomial in book
    return D, r2


R3 = [4, 3, 2]  # for c(2)x^2 + c(1)x + c(0) [c(2), c(1), c]
T, R2 = my_ploy_der_mat(R3)
print('Transformation matrix:')
print(np.stack(T), '\n')
print('''R2 Basis Vector used to show the P2' = P dx'/dy(x^2 + x + 2) = 2x''')
print(R2, '\n')

print('EX8')
# intialize a matrix and solution basis vector


def gauss_elem(A, b, n):
    a = np.zeros((n, n+1))
    x = np.zeros(n)
    print('numpy solution:')
    sol = np.round(np.linalg.solve(A, b), 8)
    val = [['x1 = '], ['x2 = '], ['x3 = ']]
    sol_ = np.hstack((val, sol))  # created a solution from linalg.solve()
    # To remove ([] , '')from evertyhing two jooins for each set of brackets
    print('\n'.join(map(lambda b: ''.join(map(str, b)), sol_)), '\n')

    a = np.hstack((A, b))  # created the augmented matrix
    print('The Augmented Matrix;')
    print(a, '\n')

    # Gauss Elimination process
    # applies the elimination below the fixed row
    for i in range(n):
        if a[i][i] == 0.0:
            sys.exit('Divide by zero detected!')

        for j in range(i+1, n):
            ratio = a[j][i]/a[i][i]

            for k in range(n+1):
                a[j][k] = a[j][k] - ratio * a[i][k]

    x[n-1] = a[n-1][n]/a[n-1][n-1]  # first x

    # substution starts here to find values of x
    for i in range(n-2, -1, -1):
        x[i] = a[i][n]

        for j in range(i+1, n):
            x[i] = x[i] - a[i][j]*x[j]
        x[i] = x[i]/a[i][i]

    c = 0
    print('\nThe solution Guass Elimination Method is: ')
    for i in range(n):
        c += 1
        print('x%d = %0.8f' %(c, x[i]))
    print('\n')
    return(x)


M = np.array([[3,  -1,  4],
              [17,  2,  1],
              [1,  12, -7]], float)
b = np.array([[2],
              [14],
              [54]], float)
n = 3
x = gauss_elem(M, b, n)

# code from the following was used and modifyed to fit the problem
# https://www.delftstack.com/howto/python/gaussian-elimination-using-pivoting/

print('EX9')
# Implementation for Gauss-Jordan Elimination Method


# Function to print the matrix
def PrintMatrix(a, n):
    for i in range(n):
        print(np.round([*a[i]], 3))


# function to reduce matrix to reduced
# row echelon form.
def PerformOperation(a, n):
    flag = 0
    # Performing elementary operations
    for i in range(n):
        if (a[i][i] == 0):
            c = 1
            while ((i + c) < n and a[i + c][i] == 0):
                c += 1
            if ((i + c) == n):
                flag = 1
                break
            j = i
            for k in range(1 + n):
                temp = a[j][k]
                a[j][k] = a[j+c][k]
                a[j+c][k] = temp

        for j in range(n):
            # Excluding all i == j
            if (i != j):
                # Converting Matrix to reduced row
                # echelon form(diagonal matrix)
                ratio = a[j][i] / a[i][i]
                k = 0
                for k in range(n + 1):
                    a[j][k] = a[j][k] - (a[i][k]) * ratio
    return flag


# Function to print the desired result
# if unique solutions exists, otherwise
# prints no solution or infinite solutions
# depending upon the input given.
def PrintResult(a, n, flag):
    c = 0
    print("Result is : ")
    if (flag == 2):
        print("Infinite Solutions Exists<br>")
    elif (flag == 3):
        print("No Solution Exists<br>")
    # Printing the solution by dividing constants by
    # their respective diagonal elements
    else:
        for i in range(n):
            x = a[i][n] / a[i][i]
            c += 1
            print('x%d = %0.8f' %(c, x))
    print('\n')
    return x


# To check whether infinite solutions
# exists or no solution exists
def CheckConsistency(a, n, flag):
    # flag == 2 for infinite solution
    # flag == 3 for No solution
    flag = 3
    for i in range(n):
        sum = 0
        for j in range(n):
            sum = sum + a[i][j]
        if (sum == a[i][j]):
            flag = 2
    return flag


def main(A, b, n):
    flag = 0
    a = np.hstack((A, b))

    # Performing Matrix transformation
    flag = PerformOperation(a, n)
    if (flag == 1):
        flag = CheckConsistency(a, n, flag)

    # Printing Final Matrix
    print("Final Augmented Matrix is : ")
    PrintMatrix(a, n)
    print()

    # Printing Solutions(if exist)
    x = PrintResult(a, n, flag)
    return x


# Array and basis vector info
A = np.array([[3, -1,  4],
              [17, 2,  1],
              [1, 12, -7]], float)
b = np.array([[2],
             [14],
             [54]], float)
n = 3  # Order of Matrix(n) or number of unknows
# Runs the main function and returns results.
x = main(A, b, n)

# This code is contributed by phasing17 and modified to fit the problem
# code from the following
# https://www.geeksforgeeks.org/program-for-gauss-jordan-elimination-method/

print('EX10')
P, L, U = lu(A)
print('A matrix:  ')
print(A, '\n')
print('Lower Triangle: ')
print(L, '\n')
print('Upper Triangle: ')
print(U, '\n')

print('EX11')
print("""
Show that dot product distributes across vector
addition: show u·(v + w)  =  (u·v) + (u·w).\n""")
u = np.array([2, 4, 6, 8, 3, 9])
print('u:', u)
v = np.array([4, 5, 6, 7, 8, 9])
print('v:', v)
w = np.array([1, 2, 3, 4, 5, 6])
print('w:', w, '\n')
dot_add = np.multiply(u, np.add(v, w))
add_dot = np.add(np.multiply(u, v), np.multiply(u, w))
print('u·(v + w)     = ', dot_add)
print('(u·v) + (u·w) = ', add_dot, '\n')
if np.sum(dot_add) == np.sum(add_dot):
    ans = 'Yes'
print('Is u·(v + w)  =  (u·v) + (u·w) ? ', ans, '\n')

print('EX12\n')
"""
I tried to set this up as with a series of linear equations but
the matrix had independent rows with determinate of 0, singular, many
solutions. Broke it down to a 3 x 3 and still had a determinate
of 0. Could not use any matrix or single vector calculation
that made any since.I broke it down into its parts and solved it.
There are any number of solutions where f6 or f7 can be assumed
which will allow for a single solution. The number assumed should be in
the range from 0 to 4 to correctly represent the flow vectors."""


def my_flow_calculator(S, d):
    f7 = 1.5  # assume a number 0-4
    f6 = d[4] - f7
    f1 = S[0]
    f2 = d[2]
    f3 = d[3] - f1 + f2
    f4 = d[0] + f3 + f6
    f5 = d[2] + f7
    f = np.array([f1, f2, f3, f4, f5, f6, f7])
    return f


S_case1 = [10, 10]
d_case1 = [4, 4, 4, 4, 4]
S_case2 = [10, 10]
d_case2 = [3, 4, 5, 4, 4]
f1 = my_flow_calculator(S_case1, d_case1)
f2 = my_flow_calculator(S_case2, d_case2)
print(f'Case 1 where, S = {S_case1} and d = {d_case1}')
print(f'flow f1 = {f1[0]}\n f2 = {f1[1]}\n f3 = {f1[2]}\n '
      f'f4 = {f1[3]}\n f5 = {f1[4]}\n f6 = {f1[5]}\n f7 = {f1[6]}\n')
print(f'Case 2 where, S = {S_case2} and d = {d_case2}')
print(f'flow\n f1 = {f2[0]}\n f2 = {f2[1]}\n f3 = {f2[2]}\n'
      f' f4 = {f2[3]}\n f5 = {f2[4]}\n f6 = {f2[5]}\n f7 = {f2[6]}\n')
