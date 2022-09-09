#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd

print('EX 1, EX 2')
print('How would you define the size of the following task.')
task_dict = {1: ['Solve a jigsaw puzzle', 'n number of pieces, sorted',
                 'O(nlog(n)'],
             2: ['Pass a handout to a class', 'n papers / n student',
                 'O(1)'],
             3: ['Walking to class', 'n distance, direction', 'O(n)'],
             4: ['Finding a name in a dictonary', 'n names, back/forward sort',
                 'O(log(n))']}
task_df = pd.DataFrame.from_dict(task_dict,
                                 orient='index')\
                                    .rename(columns={0: 'tasks',
                                                     1: 'size',
                                                     2: 'complexity'})
print(task_df, '\n')

print('EX 3')


# Function to display hashtable
def display_hash(hashTable):
    for i in range(len(hashTable)):
        print(i, end=" ")
        for j in hashTable[i]:
            print("-->", end=" ")
            print(j, end=" ")
        print()


# Creating Hashtable as
# a nested list.
HashTable = [[] for _ in range(6)]


# Hashing Function to return
# key for every value.
def Hashing(keyvalue):
    return keyvalue % len(HashTable)


# Insert Function to add
# values to the hash table
def insert(Hashtable, keyvalue, value):
    hash_key = Hashing(keyvalue)
    Hashtable[hash_key].append(value)


# Driver Code
insert(HashTable, 5, 'Dallas')
insert(HashTable, 2, 'Austin')
insert(HashTable, 6, 'Houston')
insert(HashTable, 4, 'San Antonio')
insert(HashTable, 1, 'Fort Worth')
insert(HashTable, 3, 'Midland')

display_hash(HashTable)

print("""\nThe performance of hashing is evaluated on
the basis that each key is equally likely to be
hashed for any slot of the hash table.

m = Length of Hash Table:
m = line 17 is 6 above
n = Total keys to be inserted in the hash table
n = lines 34 to 40 is 6
Load factor LF = n/m = 6/6 = 1
Expected time to search = O(1 + LF ) = O(2)
Expected time to insert/delete = O(1 + LF)

The time complexity of search insert and delete is
O(1) if  LF is O(1)\n""")

print('EX 4')
print("""For Tribonacci base case using recursion this is
very simular to the calculation for the Tower of Hanio.
eq1 T(n) = 2T(n-1)+1,
back substution
eq2 T(n-1) = 2T(n-2) _1 and eq3 T(n-2) = 2T(n-3)+1.
Substuting the value of T(n-2) into eq2 yields T(n-1) = 2(2T(n-3)+1)+1.
Substutiing the value of T(n-1) into eq1 yields T(n) = 2(2(2T(n-3)+1)+1)+1.
this simplifies to T(n) = 2^3T(n-3)+2^2+2^1 and generalized to
t(n) = 2^cT(n-c)+2^(c-1)=2^(c-2)+.......2^1+1.
for a base condition T(1)=1 n-k=1 > k=n-1 yield a sum of 2^n - 1
T(n) = O(2^n-1) ~ O(2^n).
Therefore exponential.

For the case of iterative it depends on the
iterative method used assuming bottom up ot top down the
complexity is linear O(n) using two for loops basic operations of C(n)
Therefore polynomial of order 2 \n""")

print("""For the Timmynacci with recursion it will half and the quarter
on each call n/2 + n/4 = 3n/4 next n/4 + n/8 + n/8 + n/16 = 9n/16,
this will go to n/2^c or c = log(n) for t(n)=1 if n < 1 solving the
integral int(1/u)du yields log(n) using Akra-Bazzi method yields
complexity O(n(1+log(n))) or O(nlogn)
therefore for recursion log time.

For iterative method t(n) = t(n/2) + t(n/4) appears to use a
substitution method which would run forward to a tolance.
Yielding something simular to the above with out the recursion twice using
the solution from the intergal of from n to 1 int(x/(x^p+1))dx = n^(1-p),
therefor t(n) = o(n^p(1+n^(1-p))) = n^p*(n^(1-p)) = O(n)
Iterative method is t(n) = O(n) a polynomial\n or could be calculated
to O(nlog(n) which is also linear or depending on where you start in the
list.""")
ex4_df = pd.DataFrame\
    .from_dict({'Tribonacci_Recursion': ['exponential', 'O(2^n)'],
                'Tribonacci_Iterative': ['polynomial', 'O(n^2)'],
                'Timmynacci_Recursion': ['log', 'O(nlog(n))'],
                'Timmynacci_Iterative': ['polynomial', 'O(n^c)']},
               orient='index').rename(columns={0: 'algorithm',
                                               1: 'complexity'})
print(ex4_df, '\n')

print('EX 5')

print('Tower of Hanoi given in chaper 6 is a recurive method'
      'and the complexity can be determined as follows:')
print("""For n = 3 we can use the recursive eq1 T(n) = 2T(n-1)+1,
back substution eq2 T(n-1) = 2T(n-2) _1 and eq3 T(n-2) = 2T(n-3)+1.
Substuting the value of T(n-2) into eq2 yields T(n-1) = 2(2T(n-3)+1)+1.
Substutiing the value of T(n-1) into eq1 yields T(n) = 2(2(2T(n-3)+1)+1)+1.
this simplifies to T(n) = 2^3T(n-3)+2^2+2^1 and generalized to
t(n) = 2^cT(n-c)+2^(c-1)=2^(c-2)+.......2^1+1.
for a base condition T(1)=1 n-k=1 > k=n-1 yield a sum of 2^n - 1
T(n) = O(2^n-1) ~ O(2^n), for three disk (n=3) the number of moves are
2^3-1 = 7 for n=5 31 moves n=7 127 moves n=9 511 moves\n""")

print('EX 6')

print("""Lets T(n) be the time complexity for best cases
n = total number of elements
then
T(n) = 2*T(n/2) + constant*n
2*T(n/2) is because we are dividing array into two array of equal size
constant*n is because we will be traversing elements of array in each
level of tree; therefore, T(n) = 2*T(n/2) + constant*n
further we will devide arrai in to array of equalsize so
T(n) = 2*(2*T(n/4) + constant*n/2) + constant*n == 4*T(n/4) + 2*constant*n

for this we can say that
T(n) = 2^k * T(n/(2^k)) + k*constant*n
then n = 2^k
k = log2(n)
therefore,
T(n) = n * T(1) + n*logn = O(n*log2(n))
;This is also the average case just a slight change in constants
T(n) = 2*c*log(n)*(n+1) removing constants = log(n)*(n+1) = O(nlog(n).

Worst Case happens when we will when our array will be sorted and we select
smallest or largest indexed element as pivot.

lets T(n) ne total time complexity for worst case
n = total number of elements
T(n) = T(n-1) + constant*n
as we are dividing array into two parts one consist of single element and
other of n-1 and we will traverse individual array

T(n) = T(n-2) + constant*(n-1) + constant*n =  T(n-2) + 2*constant*n - constant
T(n) = T(n-3) + 3*constant*n - 2*constant - constant
T(n) = T(n-k) + k*constant*n - (k-1)*constant ..... - 2*constant - constant
T(n) = T(n-k) + k*constant*n - constant*[(k-1) ....  + 3 + 2 + 1]
T(n) = T(n-k) + k*n*constant - constant*[k*(k-1)/2]
put n=k
T(n) = T(0) + constant*n*n - constant*[n*(n-1)/2]
removing constant terms
T(n) = n*n - n*(n-1)/2
T(n) = O(n^2)\n""")

print('EX 7')
print("""Run the following two iterative implementations of finding
Fibonacci numbers in the line_profiler as well as using the magic command
to get the repeated run time. The first implementation preallocates memory
to an array that stores all the Fibonacci numbers. The second implementation
expands the list at each iteration of the for-loop.\n""")
# %load_ext line_profiler


def my_fib_iter1(n):
    out = np.zeros(n)
    out[:2] = 1
    for i in range(2, n):
        out[i] = out[i-1] + out[i-2]
    return out


def my_fib_iter2(n):
    out = [1, 1]
    for i in range(2, n):
        out.append(out[i-1]+out[i-2])
    return np.array(out)


n = 10
print(f'for n = {n} the Fibonacci iter1 =  {my_fib_iter1(n)}')
# get_ipython().run_line_magic('time', 'my_fib_iter1(n)')
# get_ipython().run_line_magic('timeit', 'my_fib_iter1(n)')
print('\n')
print(f'for n = {n} the Fibonacci iter2 = ', my_fib_iter2(n))
# get_ipython().run_line_magic('time', 'my_fib_iter2(n)')
# get_ipython().run_line_magic('timeit', 'my_fib_iter2(n)')
print('\n')
# %lprun -f my_fib_iter1 my_fib_iter1(10)
# %lprun -f my_fib_iter2 my_fib_iter2(10)
pd.set_option('max_colwidth', 500)
pd.set_option('display.max_columns', None)
profile_df = pd.read_csv('ex8_prob7_profile.csv')
profile_df.index += 8
profile_df.fillna('', inplace=True)
print(profile_df, '\n')

# iter1
