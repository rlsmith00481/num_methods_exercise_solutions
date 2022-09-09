# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 17:18:16 2022

@author: rlsmi
"""

import pandas as pd
import numpy as np

print('EX 1')


def my_sum(lst):
    """This is a iteration function which is very straight foward and easy
    to read. It uses slice and the last sum value (s) such that
    1st s = 0 + first entry of the list
    2nd s = first entry of the list + the second entry of the list
    3rd s = sum of first and second entry + third and so on."""
    s = 0
    if len(lst) == 1:
        return lst[0]
    else:
        for i in range(len(lst)):
            s = s + lst[i]
        return s


def my_sum2(lst):
    """ This is a recursion method in the function where the sum starts with
    the last two elements in the list and works the sum to the last basically
    working backward through the list. It makes 15 calculations in 5 loops
    before it reaches the last number in the list. The iteration method only
    uses 5.
    iteration method
    0.535 µs ± 25 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each
    recursion method
    1.08 µs ± 36.4 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each
    A long list of numbers and the recursion method takes exponentially longer.
    """

    if len(lst) == 1:
        return lst[0]
    else:
        # The lst[0] has to be in the statement or it creates a
        # continues loop.
        s = lst[0] + my_sum2(lst[1:])
        # In the else: statement we will add the first element from the list
        # which is list[0] to the rest of the elements in the list.This is
        # shown by calling the function recursively with the list shorter by
        # 1 element--the element at index 0-- listsum(list[1:]), this process
        # repeats with the list getting smaller until you arrive at the base
        # case--a list of length 1 and then you will get a final result.
        # ls[0:] is equivalent to ls (well, it's ls copy to be precise)
        # As a result, listSum would always be called with same argument and
        # in result you'll reach recursion limit [it'll run infinitely].
        return s


my_lst = [1, 12, 23, 44, 85]
ss = my_sum(my_lst)
print(f'This is sum of values using iteration method {ss}.')
ss2 = my_sum2(my_lst)
print(f'This is sum of values using recursion method {ss2}.\n')

print('EX 2')


def T(n, x):
    """This is the recursive function for the Chebyshev polynomial and it is
    called by the chebyshev_poly function. In the chebyshev_poly a loop is
    formed to call the recursive function Tn(x) for the conditions where:
    n = 0 Tn(x) = 1,
    n = 1 Tn(x) = x
    n = others 2x Tn-1(x) - Tn-2(x). chebyshev_poly is used to allow a list to
    be passed into the Tn(x) function for both n and x. Due to the return
    statement in the Tn(x) function cannot immediately return the value till
    the recursive call returns a result. it would be necessary modify the
    passed as only a index """
    if n == 0:
        return 1
    elif n == 1:
        return x
    else:
        return 2. * x * T(n - 1, x) - T(n - 2, x)


def chebyshev(n, x):
    cas1 = []
    cas2 = []
    cas3 = []
    cas4 = []

    for i in x:
        c1 = T(n[0], i)
        cas1.append(c1)
        c2 = T(n[1], i)
        cas2.append(c2)
        c3 = T(n[2], i)
        cas3.append(c3)
        c4 = T(n[3], i)
        cas4.append(c4)
    return cas1, cas2, cas3, cas4


x = [1, 2, 3, 4, 5]
n = [0, 1, 3, 4]
case1, case2, case3, case4 = chebyshev(n, x)
print('The Chebyshev polynomial for n = 0, 1, 3\n', 'n = 0: ', case1, '\n',
      'n = 1: ', case2, '\n', 'n = 3: ', case3, '\n', 'n = 4: ', case4, '\n')
dict_1 = {'n = 0': case1, 'n = 1': case2, 'n = 3': case3, 'n = 4': case4}
EX2_df = pd.DataFrame(dict_1, index=x)
EX2_df.index.name = 'Values'
pd.set_option('display.colheader_justify', 'center')
print(EX2_df, '\n')

print('EX 3')


def A(m, n):
    if m == 0:
        return n + 1
    if n == 0:
        return A(m - 1, 1)
    return A(m - 1, A(m, n - 1))


case1 = A(1, 1)
case2 = A(1, 2)
case3 = A(2, 3)
case4 = A(3, 3)
case5 = A(3, 4)
print('The Ackermann function for:\n', 'm=1,n=1: ', case1, '\n',
      'm=1,n=2: ', case2, '\n', 'm=2,n=3: ', case3, '\n',
      'm=3,n=3: ', case4, '\n', 'm=3,n=4: ', case5, '\n')
dict_A = {'m': [1, 1, 2, 3, 3], 'n': [1, 2, 3, 3, 4],
          'Results': [case1, case2, case3, case4, case5]}
EX3_df = pd.DataFrame(dict_A, index=['case1', 'case2', 'case3',
                                     'case4', 'case5'])
print(EX3_df)

print('EX 4')


def C(n, k):
    if n == k:
        return 1
    if k == 1:
        return n
    return C(n - 1, k) + C(n - 1, k - 1)


n = [10, 10, 10, 100]
k = [1, 10, 3, 3]
ex1 = C(n[0], k[0])
ex2 = C(n[1], k[1])
ex3 = C(n[2], k[2])
ex4 = C(n[3], k[3])
print('The Choose function for:\n', 'n=10,k=1: ', ex1, '\n', 'n=10,k=10: ',
      ex2, '\n', 'n=10,k=3: ', ex3, '\n', 'n=100,k=3: ', ex4, '\n')
dict_C = {'n': n, 'k': k, 'Results': [ex1, ex2, ex3, ex4]}
EX4_df = pd.DataFrame(dict_C, index=['ex1', 'ex2', 'ex3', 'ex4'])
print(EX4_df,'\n')


print('EX 5')


def my_change(cost, paid):
    """The outer function my_change(cost, paid) creates the data for the enter
    function return_change. The inputs are the cost of a product and the amount
    paid. The recursive function is return_change(to_return). It produces a
    nested list of the coinage used such as the following:
   [50.0, [20.0, [1.0, [1.0, [0.25, [0.1, [0.05, [0.01, [0.01, [0.01]]]]]]]]]],
    where paid = $100.0 and cost = $27.57 differance of $72.43.
    The enter function flatten use an iterator L for an item but in this case
    the items are alread an item therefore a type error occures yielding the
    item creating a list of items form the the nested list. """
    change = paid - cost

    def return_change(to_return):
        """# YouTube Walk Through:
        https://www.youtube.com/watch?v=8JwdenBGmEo&feature=youtu.be"""
        coins = [.01, .05, .10, .25, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0]
        flag = None
        for c in coins:
            if c == to_return:
                return [c]
            if c < to_return:
                flag = c
        temp_balance = round(to_return - flag, 2)
        return [flag] + [return_change(temp_balance)]

    def flatten(L):
        """Recursive function to flatten an iterable
        with arbitrary levels of nesting.
        https://stackoverflow.com/a/14491059/3182843"""
        for item in L:
            try:
                yield from flatten(item)
            except TypeError:
                yield item
    rt_ch = return_change(change)
    print(f'This is the recursion process for the change;\n {rt_ch}')
    return flatten(rt_ch)


invoice = 27.57
cash = 100.00
result = my_change(invoice, cash)
result = list(result)
print(f'This is the listing of the change values;\n {result}')
change_back = round(sum(result), 2)
check = change_back + invoice
print('The change back + billed amount = cash recived.')
print(f'\t{change_back}\t    +  \t{invoice}\t    =  \t{check}\n')

print('EX 6')
# Enter (n) the number of iterations to find the Golden Ratio.
n = 20


def my_golden_ratio1(n):
    """This function will calculate the golden ratio without using any
    recursion methods. The following uses the relationship between the
    fibonacci number generator in n number excluding [0, 1]:
    fibonacci number(n) /fibonacci number(n-1)
    starting at n = 1 excluding the first 2 numbers. """
    fiblist = [0, 1]
    for i in range(n):
        fiblist.append(fiblist[i] + fiblist[i+1])
    print(f'The fibonacci list of numbers excluding [0, 1]: {fiblist}.')
    gratio = [fiblist[i] / float(fiblist[i-1]) for i in range(2, len(fiblist))]

    return gratio[n-1]


print(f'This is the golden ratio without any recursive method being applied:'
      f' {round(my_golden_ratio1(n),6)}.')


def my_golden_ratio2(x):
    """This function uses recursion method to calculate the fibonacci number(n)
    and then iterates through the  fibonacci number(x) /fibonacci number(x-1)
    starting at n = 1 excluding the first 2 numbers. This is the same process
    used abouve with the exception of the enter function which
    caculates the  fibonacci number(n) using recursion method."""
    fiblist = [0, 1]

    def fibonacci(n):
        if n == 0:
            return 0
        elif n == 1:
            return 1
        return fibonacci(n-1) + fibonacci(n-2)

    for i in range(1, x+2):
        fiblist.append(fibonacci(i))
    gratio = [fiblist[i] / float(fiblist[i-1]) for i in range(2, len(fiblist))]

    return gratio[x]


print(f'This is using the recursive method for fibonacci number to obtian the'
      f'golden ratio: {round(my_golden_ratio2(n),6)}.')


def my_golden_ratio(n):
    """This function calculates the golden ratio using fraction recusion
    relationship. The sequence is:
     Σ(n)  1 + 1/Φ(n) where n is the number of recursions."""
    if n == 0:
        return 1
    else:
        for i in range(n):
            grat = 1 + 1/my_golden_ratio(i)

    return grat


gr = my_golden_ratio(n)
print(f'This is the fraction recursive method  of the golden ratio:'
      f'{round(gr, 6)}.')
print(f'This is the Fibonacci approximation for the golden ratio:'
      f'{round((1+ np.sqrt(5))/2,6)}.')
print('All the methods above using 19 iterations will come within 6 decimal'
      ' points of the golden ratio.')
print(f'The wide-screen display has a ratio of 16:9 or {round(16/9, 6)}.\n')

print('EX 7')


def my_gcd(a, b):
    r = a % b
    if(r == 0):
        return b
    else:
        return my_gcd(b, r)


x = [10, 33, 18]
y = [4, 121, 1]
gcd_list = []
for i in range(len(x)):
    cs = my_gcd(x[i], y[i])
    gcd_list.append(cs)
print(f'the greatest common divisor for numerators\n {x}\n and denominators\n'
      f' {y}\n yields\n {gcd_list}.')
my_gcd_df = pd.DataFrame(list(zip(x, y, gcd_list)),
                         columns=['x', 'y', 'divisor'],
                         index=['case 1', 'case 2', 'case 3'])
pd.set_option('display.colheader_justify', 'center')
print(my_gcd_df, '\n')

print('EX 8\n')


def my_pascal_row(m):
    """Recursive function to calculate elemnets on the Pascals Triangle
    for m number of rows. The following code prints out Pascal's triangle
    for a specified number of rows. Pascals triangle is a triangle of the
    binomial coefficients. The values held in the triangle are generated
    as follows:
    In row 0 (the topmost row), there is a unique nonzero entry 1.
    Each entry of each subsequent row is constructed by adding the number
    above and to the left with the number above and to the right, treating
    blank entries as 0. For example, the initial number in the first
    (or any other) row is 1 (the sum of 0 and 1), whereas the numbers 1 and 3
    in the third row are added together to generate the number 4 in the fourth
    row."""
    if m == 0:
        return []

    elif m == 1:
        return [[1]]  # Base case termination condition

    else:
        pas = [1]  # Calculate current row using data from previous row
        tri_lst = my_pascal_row(m-1)  # Recursive
        last_elem = tri_lst[-1]  # Take from end of result
        for i in range(len(last_elem) - 1):
            pas.append(last_elem[i] + last_elem[i + 1])
        pas += [1]
        tri_lst.append(pas)
    return tri_lst


"""Note that the Fibonacci is the sum of the number on diagonal, using this
one could construct a encription using a pyramid shape."""

lst = (my_pascal_row(8))

for i in range(len(lst)):
    print(f'row number {i+1} {lst[i]}')
print('')


tri_array = my_pascal_row(8)
# Prints the pascal triangle.
for i, row in enumerate(tri_array):
    for j in range(len(tri_array) - i + 1):
        print(end="    ")  # leading spaces.
    for j in row:
        print(j, end="      ")  # print entries.
    print("\n")  # print new line

print('EX 9 and 10')
"""Could not figure this one out will have to revisit this problem.
I was able to get the infomation that I need to understand the sprial matrix
but could not complete the problem as stated."""


def spiral(R, C):
    grid = [[0] * C for _ in range(R)]
    directions3 = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    # directions2 = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    # directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    dirIdx = 0
    count = 0
    zo = str(0)
    r, c = 0, 0

    for i in range(R*C, 0, -1):
        count += 1

        grid[r][c] = zo

        if r == R//2 and c == R//2:
            grid[r][c] = 1
        elif r == 0 and c == 0:
            grid[r][c] = 1
        elif r == R-1 and c < C-1:
            grid[r][c] = 1
        elif r >= R-3 and c == C-2:
            grid[r][c] = 1

        nr, nc = r+directions3[dirIdx][0], c+directions3[dirIdx][1]
        # print('new1', nr-r, nc-c)
        if nr < 0 or nr >= R or nc < 0 or nc >= C or grid[nr][nc] != 0:
            dirIdx = (dirIdx + 1) % 4
            nr, nc = r + directions3[dirIdx][0], c + directions3[dirIdx][1]
            # print('new2', nr-r, nc-c)
        r, c = nr, nc
    return grid


s = spiral(5, 5)
for row in s:
    print(row)


def spiral(n):
    def spiral_part(x, y, n):
        if x == -1 and y == 0:
            return -1
        if y == (x+1) and x < (n // 2):
            # print(x-1, y-1, n-1,4*(n-y))
            # print(spiral_part(x-1, y-1, n-1) )
            return spiral_part(0, 0, n) + 0*(n-y)
        if x < (n-y) and y <= x:
            # print(x-1, y, n, x-y))
            # print(spiral_part(x-1, y-1, n-1))
            return spiral_part(y-1, y, n) + (1) + 1
        if x >= (n-y) and y <= x:
            return spiral_part(x, y-1, n) + 1
        if x >= (n-y) and y > x:
            return spiral_part(0, 0, n) + 0
        if x < (n-y) and y > x:
            return spiral_part(0, 0, n)

    array = [[0] * n for j in range(n)]
    for x in range(n):
        for y in range(n):
            array[x][y] = spiral_part(y, x, n)
    return array


for row in spiral(5):
    print(" ".join("%2s" % x for x in row))
print('\n\n')


print('EX 12 EX 13 Hanoi Tower with out recursion.')
# Python3 program for iterative Tower of Hanoi
import sys  # this should be at the top of the script left here for easy access
direct = []  # global variable

# A structure (blueprint) to define the group
# disk stacks and its functions.


class Stack:
    # Constructor to set the data of
    # the newly created tree node
    # capacity is the number of disk,
    def __init__(self, capacity):
        self.capacity = capacity
        self.top = -1
        self.array = [0]*capacity

# function to create a stack of given capacity.
# How many disk are in the stack of disk.
# number_of_disk


def createStack(capacity):
    stack = Stack(capacity)
    return stack

# Stack is full when top is equal to the last index
# never gets full until the last disk is placed.


def isFull(stack):
    return (stack.top == (stack.capacity - 1))

# Stack is empty when top is equal to -1 and
# this occures at least once during the movement.


def isEmpty(stack):
    return (stack.top == -1)

# Function to add a disk to stack.
# It increases top by 1 item is the disk.
# Again the stack is never full until the
# last disk is placed once movement starts.


def push(stack, item):
    if(isFull(stack)):
        return
    stack.top += 1
    stack.array[stack.top] = item

# Function to remove a disk from stack.
# It decreases top by 1. its called to
# move the next disk.


def Pop(stack):
    if(isEmpty(stack)):
        return -sys.maxsize
    Top = stack.top
    stack.top -= 1
    return stack.array[Top]

# Function to implement legal
# movement between two poles
# src is the source pole and dest is the destion pole
# to remove from and s and d are polls they went to.


def moveDisksBetweenTwoPoles(src, dest, s, d):
    pole1TopDisk = Pop(src)  # calls the remove disk from stack
    pole2TopDisk = Pop(dest)

    # When pole 1 is empty
    if (pole1TopDisk == -sys.maxsize):
        push(src, pole2TopDisk)
        moveDisk(d, s, pole2TopDisk)

    # When pole2 pole is empty
    elif (pole2TopDisk == -sys.maxsize):
        push(dest, pole1TopDisk)
        moveDisk(s, d, pole1TopDisk)
        # at no time will there be 3 empty poles in a 3 pole system.
        # When top disk of pole1 > top disk of pole2
    elif (pole1TopDisk > pole2TopDisk):
        push(src, pole1TopDisk)
        push(src, pole2TopDisk)
        moveDisk(d, s, pole2TopDisk)

    # When top disk of pole1 < top disk of pole2
    else:
        push(dest, pole2TopDisk)
        push(dest, pole1TopDisk)
        moveDisk(s, d, pole1TopDisk)


# Function to show the movement of disks
def moveDisk(fromPeg, toPeg, disk):

    lst = [disk, fromPeg, toPeg]
    direct.append(lst)

    print("Move disk", disk, "from tower'", fromPeg, "' to tower'", toPeg, "'")
    return direct

# The function to implement the total process.
# The final pole will be pole B for even number
# of disks and pole C for odd for. To move to change the order
# change the pole order change d and a locations below.


def tohIterative(num_of_disks, src, aux, dest):
    s, d, a = 'A', 'B', 'C'  # could changed names to pole 1, 2, 3

    # If number of disks is even, then interchange
    # destination pole and auxiliary pole
    if (num_of_disks % 2 == 0):
        temp = d
        d = a
        a = temp
    total_num_of_moves = int(pow(2, num_of_disks) - 1)

    # Larger disks will be pushed first
    for i in range(num_of_disks, 0, -1):
        push(src, i)

# Move disk between poles tracking the total number of moves
# starting at 1, where the total number of moves is 2^n - 1.
    for i in range(1, total_num_of_moves + 1):
        if (i % 3 == 1):
            moveDisksBetweenTwoPoles(src, dest, s, a)  # Print state included.

        elif (i % 3 == 2):
            moveDisksBetweenTwoPoles(src, aux, s, d)

        elif (i % 3 == 0):
            moveDisksBetweenTwoPoles(aux, dest, d, a)


# Input: number of disks
num_of_disks = 3
n = num_of_disks
idx = np.arange(1, 2**n, 1)

# Create three stacks of size 'num_of_disks'
# to hold the disks
src = createStack(num_of_disks)  # pole 1
aux = createStack(num_of_disks)  # pole 2
dest = createStack(num_of_disks)  # pole 3

# Function to shuffle disk
tohIterative(num_of_disks, src, aux, dest)
Hanoi_df = pd.DataFrame(direct,
                        columns=['Disk', 'From Tower', 'To Tower'], index=idx)
Hanoi_df.index.name = 'Steps'
pd.set_option('display.colheader_justify', 'center')
print(Hanoi_df,'\n')
# This code is contributed by divyeshrabadiya07

print('EX 12 EX 13 my_quicksort with out recursion.')


def my_quicksort_a(lst):
    """First run through the function set the pivot for smaller and bigger
    numbers, The next time it uses the smaller list and then bigger
    to find all the smaller numbers in the list and bigger ones. This
    continues until all numbers are sorted using the new povot number.
    programed for assending order"""
    if len(lst) <= 1:
        sorted_list0 = lst
    else:
        pivot = lst[0]
        bigger = []
        smaller = []
        same = []
        for item in lst:
            if item > pivot:
                bigger.append(item)
            elif item < pivot:
                smaller.append(item)
            else:
                same.append(item)
        sorted_list0 = my_quicksort_a(smaller) + same + my_quicksort_a(bigger)

    return sorted_list0


def my_quicksort2(lst2):
    """This function does almost the same with the exception of not calling the
    function for an iteration. This function does both assending and desending
    ored for the sort. """
    sorted_list_a = []
    sorted_list_d = []
    for num in lst2:
        sorted_list_a = [item for item in sorted_list_a if num > item] + [num]\
            + [item for item in sorted_list_a if num <= item]
        sorted_list_d = [item for item in sorted_list_d if num <= item] + [num]\
            + [item for item in sorted_list_d if num > item]

    return sorted_list_a, sorted_list_d


def my_quicksort_d(lst):
    """First run through the function set the pivot for smaller and bigger
    numbers, The next time it uses the smaller list and then bigger
    to find all the smaller numbers in the list and bigger ones. This
    continues until all numbers are sorted using the new povot number.
    programed for desending order."""
    if len(lst) <= 1:
        sorted_list1 = lst
    else:
        pivot = lst[0]
        bigger = []
        smaller = []
        same = []
        for item in lst:
            if item < pivot:
                bigger.append(item)
            elif item > pivot:
                smaller.append(item)
            else:
                same.append(item)
        sorted_list1 = my_quicksort_d(smaller) + same + my_quicksort_d(bigger)

    return sorted_list1


my_list = [5, 4, 8, 2, 9, 1, 7, 3]

sorted2, sorted3 = my_quicksort2(my_list)
print('Using the iteration method.')
print(f'In accending order iteration method:\n {sorted2}')
print(f'In decending order iteration method:\n {sorted3}\n')

sorted0 = my_quicksort_a(my_list)
print('Using the recursion method.')
print(f'In accending order recursion method:\n {sorted0}')
sorted1 = my_quicksort_d(my_list)
print(f'In decending order recursion method:\n {sorted1}')
