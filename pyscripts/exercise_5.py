#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
from math import* 
import pandas as pd
from random import randint

print('EX 1')

"""What is the value of y after the following code is excuted?
The values is 1000 or the minimum of n or m."""
y = 0
m = 1000
n = 1000
for i in range(m):
    for j in range(n):
        if i == j:
            y += 1
print('This is the minimum number of loop of eather i or j '
      'where both match.', y, '\n')

print('EX 2')


def my_max(x):
    Max = x[0]  # First item in the list.
    print('Init Max', Max)
    for num in x:
        if Max < num:  # Test number against inital number in list.
            Max = num
    return Max


def my_max_sort(x):
    return sorted(x)[-1:]


num_lst = np.array([20, 45, 98, 32, 46, 102, 104, 10, 8])
print(f'The maximum number in the list {num_lst} is'
      f' {my_max(num_lst)} using for and if loop methods.')
print(f'The maximum number in the list {num_lst} is'
      f' {my_max_sort(num_lst)} using for sort and slice methods.\n')

print('EX 3')


def my_n_max_sort(ab, num):
    return sorted(ab)[-num:]


def my_n_max(x, n):
    count = 0
    out = []
    while count < n:
        a = max(x)
        out.append(a)
        x.remove(a)
        count += 1
    return out


# This list is for the while loop method wich uses remove list item function.
z = [7, 9, 10, 5, 8, 3, 4, 6, 2, 1]

# This is a separt list for the sort function if not then the sort function'
#  must be called first.
zz = [7, 9, 10, 5, 8, 3, 4, 6, 2, 1]
no = 3

# This function call must come first or there nneeds to be two different array
# for each  function call.
# This happens because we are removing the maximum values from the list and
# appending them to a new list.
# This modified list is past out of the function call with the three maximum
# number removed.
out = my_n_max(z, no)
# display(out)
out_sort = my_n_max_sort(zz, no)
# display(out_sort)

print(f'The {no}rd largest number from the list {z} are {out}.')
print(f'The {no}rd largest number from the list {z} are {out_sort} using the'
      ' sort and strip methods.\n')

print('EX 4')


def my_trig_odd_even(M):
    Q = np.zeros(M.shape)  # Set the Q array to write to.
    for i in range(len(M)):  # Loop through each element of the list.
        for j in range(len(M)):
            if M[i][j] % 2 == 0:  # Test for even numbers else odd.
                Q[i][j] = np.sin(M[i][j])
            else:
                Q[i][j] = np.cos(M[i][j])
    return Q


A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
q = my_trig_odd_even(A)
print('The Original Array')
print(A, '\n')
print('The even numbers are sin() and odd are cos() from the original array.')
print(q, '\n')

print('EX 5')


def my_mat_mult(P, Q):
    """If vectors are identified with row matrices, the dot product can also
    be written as a matrix product.
           P·Q = PQ.T
    where the sum of the product is the dot product.
    In order to correct for size the arrays vectors must be such that size
    PxQ 2x4 X 4x3 = 2x3, where the empty array or zero array is the correct
    size to accept the values.
    For example, a 2 × 4 matrix (row vector) is multiplied by a 4 × 3 matrix
    (column vector) to get a, 2 × 3 matrix that is identified with its
    unique entry."""
    Q = Q.T
    M = np.zeros((len(P), len(Q)))
    for i in range(len(P)):
        M[i] = sum(P[i]*Q[i])
    return M


Po = np.ones((3, 3))
Ones = my_mat_mult(Po, Po)
P1 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
Q1 = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])
PQ1 = my_mat_mult(P1, Q1)
print(f'The dot product for Po·Po where Po is \n{Po}\n')
print('Dot Product for Po·Po')
print(Ones, '\n')
print(f'The dot product for P1·Q1 where P1 is \n{P1} \nand Q1 is \n{Q1}\n')
print('Dot Product P1·Q1')
print(PQ1, '\n')

print('EX 6')


def my_saving_plan(PO, i, goal):
    """PO is the principle balance, i is the annual intrest rate, and goal is
    principle + intrest.
    n is the number of time the interest is applied per time period.
    In the case intrest is compounded annually, therefore n is equal to 1.
    the formula is; goal = PO(1 + i/n)^(nt). The time to reach a goal the
    equation can be solved for time; t = ln(goal/PO)/(n(ln(1 + i/n))).
    The answer t can be rouned up for intger number as the actual compounding
    occures annually."""
    t = (log(goal/PO)) / (1*(log(1 + i)))
    return ceil(t), t  
# I have inclued the time to accumulate and the time to with draw.


def my_saving_plan_2(PO, i, goal):
    """This is a approximation and is not correct mathematically in the time
    (t) frame would be greater than the goal and does not in any way create a
    match to the goal but only shows when that goal was reached. The formula
    Pn = (1+i)P(n-1) assumes annual compounding. The actual formula is
    Pn = (1+i)^t(P(n-1)) for annual componding."""
    P = PO
    t = 0
    while P < goal:
        P = (1 + i)*P
        t += 1
    return(t)


round_up1, yrs1 = my_saving_plan(1000, 0.05, 2000)
round_up2, yrs2 = my_saving_plan(1000, 0.07, 2000)
round_up3, yrs3 = my_saving_plan(500, 0.07, 2000)

cas1 = my_saving_plan_2(1000, 0.05, 2000)
cas2 = my_saving_plan_2(1000, 0.07, 2000)
cas3 = my_saving_plan_2(500, 0.07, 2000)
plan = pd.DataFrame({'Actual_time': [yrs1, yrs2, yrs3],
                     'Time_Rounded_Up': [round_up1, round_up2, round_up3],
                     'Aprox_time': [cas1, cas2, cas3]},
                    index=['(P=1000, i=0.05, goal=2000)',
                           '(P=1000, i=0.07, goal=2000)',
                           '(P=500, i=0.07, goal=2000)'])
plan.index.name = '(Principle, Intrest, Accured)'
print('This is the results from two different function one that calculates '
      'the time and one that approximates time to goal. The time is in years.')
pd.set_option('display.colheader_justify', 'center')
plan['Actual_time'] = plan['Actual_time'].round(2)
print(plan, '\n')

print('EX 7')


def my_find(M):
    col_idx = []
    row_idx = []

    col = int(M.size/len(M))
    for i in range(len(M)):
        row_idx.append('row')
        row_idx.append(i)
        col_idx.append('index-')
        for j in range(col):
            if M[i][j] == 1:
                col_idx.append(j)
    return row_idx, col_idx


M = np.array([[1, 0, 1, 1, 0]])
row, col = (my_find(M))
print('Row(s) with columns indexes equal to 1.')
print(row, col, '\n')

print('EX 8')


def main():
    s = ''
    while not s or s[0] in 'Yy':
        play_the_game()
        s = input('Play again? (Y or N): ')


def play_the_game():
    d1, d2, dt = roll()
    r = d1 + d2
    if r == 7 or r == 11:
        print(r, 'is an instant WINNER!\n')
        return
    if r == 2 or r == 3 or r == 12:
        print(r, 'is an instant LOSER. Sorry.\n')
        return
    print('Your point is now a', r)
    point = r
    while True:
        s = input('Roll again (E = exit)?')
        if len(s) > 0 and s[0] in 'Ee':
            return
        d1, d2, dt = roll()
        if dt != 0:
            r = dt + d1 + d2
        else:
            r = d1 + d2
        print('You rolled a', r)
        if r == point:
            print('You\'re a WINNER!\n')
            return
        elif r == 7:
            print('Sorry, you\'re a LOSER.\n')
            return


def roll():
    d1 = randint(1, 6)
    d2 = randint(1, 6)
    print(d1, d2)
    if d1 == d2:
        dt = d1 + d2
    else:
        dt = 0
    return d1, d2, dt


main()

print('EX 9')


def is_prime(n):
    prime = []
    for i in n:
        c = 0
        for j in range(2, i+1):
            if i % j == 0:
                c += 1
        if c == 1:
            prime.append(i)
    return prime


def is_prime2(m):
    pri = []
    for i in m:
        c = 0
        for j in range(1, i):
            if i % j == 0:
                c += 1
        if c == 1:
            pri.append(1)
        else:
            pri.append(0)
    return pri


prime_lst2 = is_prime2(num_lst)
prime_df = (pd.DataFrame(num_lst, index=prime_lst2, columns=['No. List']))
prime_df.index.name = 'Prime=1'
prime_df.sort_values(by='No. List', inplace=True)
num_lst = [11, 13, 5, 2, 24, 29, 8]
pd.set_option('display.colheader_justify', 'left')
print(f'The number list to find primes:\n {num_lst}\n')
prime_lst = is_prime(num_lst)
print(f'The prime numbers are:\n {prime_lst}\n')
print(f'The prime numbers in the number list\n are marked'
      f' with a 1 and\n non-prime with a 0:\n {prime_lst2}\n')
print(prime_df, '\n')

print('EX 10')


def prime_no(i, primes):
    for prime in primes:
        if not (i == prime or i % prime):
            return False
    primes.add(i)
    #  print(primes, i)
    return i


def Primes(n):
    primes = set([2])
    i, p = 2, 0
    while True:
        if prime_no(i, primes):
            p += 1
            if p == n:
                return primes
        i += 1


n = 10  # n is the number of primes to find starting at 1.
print(f'Find the first {n} prime numbers starting at 1.')
print(sorted(Primes(n)), '\n')

print('EX 11')


def is_prime(n):
    prime = []
    for i in n:
        c = 0
        for j in range(2, i+1):
            if i % j == 0:
                c += 1
        if c == 1:
            prime.append(i)
    return prime


def fibonacci_primes(n):
    fib_lst = []
    a, b = 0, 1
    c, d = 0, 1
    count = n-1

    while n > 0:
        a, b = b, a + b
        fib_lst.append(a)
        n -= 1
    n = count+1
    primes = is_prime(fib_lst)
    return fib_lst, primes


n = 30
fib_lst, fib_primes = fibonacci_primes(n)
print(f' For {n} fibonacci numbers:\n {fib_lst}.\n\n'
      f'There are {len(fib_primes)} primes in list'
      f'they are:\n {fib_primes}.\n')

print('EX 12')


def my_trig_odd_even(M):
    Q = np.zeros(M.shape)  # Set the Q array to write to.
    for i in range(len(M)):  # Loop through each element of the list.
        for j in range(len(M)):
            if M[i][j] % 2 == 0:  # Test for even numbers else odd.
                Q[i][j] = np.cos(pi/M[i][j])
            else:
                Q[i][j] = np.sin(pi/M[i][j])
    return Q


A = np.array([[3, 4], [6, 7]])
q = my_trig_odd_even(A)
print('The Original Array')
print(A, '\n')
print('The even numbers are sin() and odd are cos() from the original array.')
print(q, '\n')

print('EX 13')
data_lst = np.array(
    [[0, 1, 0, 1],
     [1, 0, 0, 1],
     [0, 0, 0, 1],
     [1, 1, 1, 0]]
    )
cities_lst = ['Los Angeles', 'New York', 'Miami', 'Dallas']


class Node:
    def __init__(self, dataval=None):
        self.dataval = dataval
        self.nextval = None
# It is a reseved method in python classes. It is known as a constructor
# in object oriented concepts.
# This method called when an object is created from the class
# and it allow the class to initialize the attributes of a class.


class SLinkedList:
    def __init__(self, nodes=None):
        self.headval = None
        if nodes is not None:
            node = Node(dataval=nodes.pop(0))
            self.headval = node
            for elem in nodes:
                node.nextval = Node(dataval=elem)
                node = node.nextval

    def listprint(self):
        printval = self.headval
        while printval is not None:
            print(printval.dataval, '\n')
            printval = printval.nextval

    def __repr__(self):
        node = self.headval
        nodes = []
        while node is not None:
            nodes.append(node.dataval)
            node = node.nextval
        nodes.append("None")
        return str(nodes)
# From the class Node class SLinkedList is used to create, print, and display
# a linked list from data in the Node function. The repr passes a str value
# not a list.


def my_connectivity_mat_2_dict(C, names):
    """ To separte out the column indexs for each last value of 1 in the
    matrix, the following was variables were isolated to create the dictionary
    at each last data point of 1 in each row. """
    node = []
    idx0 = []
    idx1 = []
    idx2 = []
    idx3 = []
    dict1 = {}
    dict2 = {}
    dict3 = {}
    dict4 = {}
    row, col = C.shape  # The row and column counts.

    for i in range(row):
        for j in range(col):
            if C[i][j] == 1 and i == 0:  # To meet the condions for each row.
                idx0.append(j+1)
                dict1 = {names[i]: idx0}
            elif C[i][j] == 1 and i == 1:
                idx1.append(j+1)
                dict2 = {names[i]: idx1}
            elif C[i][j] == 1 and i == 2:
                idx2.append(j+1)
                dict3 = {names[i]: idx2}
            elif C[i][j] == 1 and i == 3:
                idx3.append(j+1)
                dict4 = {names[i]: idx3}

    node1 = {**dict1, **dict2, **dict3, **dict4}  # Join all the dict.
    node2 = list(node1.items())  # Convet dict to a list for linked list.
    idx = [idx0, idx1, idx2, idx3]  # Create a single list of data points used.
    node = SLinkedList([node2])  # Create a liked list from converted dict.

    return node, node1  # Returned both the linked list and dictionary.


# This is a linked list of the dictionary keys and values which were converted
# to a list for the class function used to make the linked list.
print('This is the linked list of dictionary keys and values.')
con_lst, node1 = my_connectivity_mat_2_dict(data_lst, cities_lst)
# The above linked list print function from the above function.
con_lst.listprint()
# This is a print out of a dictionary separting the key and the values.
# The dictionalary was created in the function my_connectivity_mat_2_dict.
print('This is the dictionary print to look like the values in the book.\n'
      'The dictionary is not a node connection but can be used as one.')
for k, v in node1.items():
    print('node', [k], '=', v)
print('\n')

print('EX 14')
words = ['test', 'data', 'analyze']
up = [word.upper() for word in words]

print(f'All charaters in list {words} converted to upper case is {up}.')
