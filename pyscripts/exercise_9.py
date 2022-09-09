#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import struct
from itertools import product

print('EX 1')


def my_bin_2_dec(b):
    """The binary input must be a 2d list or array of a single item
    [[]] or multiple items[[],[]]"""
    d_list = []  # intialize the output list
    for item in b:
        csum = 0
        bi = item
        # bi is the 1D list in the 2D array
        for i in range(len(bi)):
            c = bi[len(bi)-i-1]*(2**(i))
            # the array is read back to for the calculations.
            csum += c
            # Sums the 2^n to calculate the decimalbase 10.
        d_list.append(csum)  # appends each caculated number
    return d_list


b = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
     [1, 0, 1, 0, 1, 0, 1],
     [1]*25, [1, 1, 1, 0, 0, 1]]
idx = [str(x) for x in b]
dec_list = my_bin_2_dec(b)
ex1_df = pd.DataFrame(dec_list,
                      index=idx,
                      columns=['decimal'])
ex1_df.index.name = 'binary'
print(ex1_df, '\n')

print('EX 2')


def my_dec_2_bin(d):
    """This function uses the bin function to make calculation.
    The input can be a tuple or list for d."""
    b_list = []
    # If a single integer bace 10.
    if isinstance(d, int):
        # Uses the bin() to calculate the binary.
        bin_int = int(bin(d)[2:])
        res = [list(map(int, str(bin_int)))]
        # Creates a list of string numbers.
        return res
    # if not a single integer create the binary number list
    for item in d:
        bin_int = int(bin(item)[2:])
        b_list.append(bin_int)
    return b_list


d = (43, 23, 2097)  # Input can be tuple or list.
bin_list = my_dec_2_bin(d)
ex2_df = pd.DataFrame(bin_list, index=d, columns=['binary'])
ex2_df.index.name = 'decimal'
print(ex2_df, '\n')


def my_dec_2_bin2(dd):
    """This function does not use the bin() for calculation.
    The calls on a another function to do the calculations
    dec_2_bin() then those are stored in list and printed in
    the forma of pands dataframe."""
    b_lst = []
    for item in dd:
        bb = dec_2_bin(item)  # calls the function dec_2_bin().
        b_lst.append(bb)
    return b_lst


def dec_2_bin(d):
    bit = []
    if d == 0:  # check to see if number is 0.
        return [0]
    while d:  # Use a while loop to make calculatins.
        bit.append(int(d % 2))
        d = d//2  # check to see the loop is contnues
    return bit[::-1]


dd = (43, 23, 2046)
bit_list = my_dec_2_bin2(dd)
print('Decimal Numbers: ', list(dd))
print('Binary List respectively: ', bit_list, '\n')

ex2_df2 = pd.DataFrame(bit_list, index=dd).fillna('')
ex2_df2.iloc[2, :] = ex2_df2.iloc[2, :].astype(int)
ex2_df2.iloc[1, :5] = ex2_df2.iloc[1, :5].astype(int)
ex2_df2.index.name = 'decimal /2^'
print(ex2_df2, '\n')

print('EX 3')


def floatToBinary64(value):
    getBin = lambda x: x > 0 and str(bin(x))[2:] or "-" + str(bin(x))[3:]
    val = struct.unpack('Q', struct.pack('d', value))[0]
    result = getBin(val)
    if value > 0:
        result = result.zfill(64)
    return result


def binaryToFloat(value):
    hx = hex(int(value, 2))
    return struct.unpack("d", struct.pack("q", int(hx, 16)))[0]


di = 43
bini = floatToBinary64(di)
conv = binaryToFloat(bini)
print(f'The base 10 number is {di} and the '
      f'binary number is {bini} and nesting the function'
      f' I get back the original {conv}\n')

print('EX 4')


def my_bin_adder(b1, b2):
    max_len = max(len(b1), len(b2))
    b1 = b1.zfill(max_len)  # zfill Fills empty spaces with 0
    b2 = b2.zfill(max_len)

    # Initialize the result
    result = ''
    res_lst = []
    # Initialize the carry
    carry = 0

    # Traverse the string
    for i in range(max_len - 1, -1, -1):  # Step backward starting at 6
        r = carry
        r += 1 if b1[i] == '1' else 0
        r += 1 if b2[i] == '1' else 0

        result = ('1' if r % 2 == 1 else '0') + result  # adding two strings
        res_lst.append(result)
        # Compute the carry.
        carry = 0 if r < 2 else 1

    if carry != 0:
        result = '1' + result
    res_lst.append(result.zfill(max_len))
    sol = res_lst[max_len]
    return res_lst, sol


def bin_add(bin_lst):
    # Initialize the flag(ck) and result.
    sol_list = []
    ck = len(bin_lst)
    # for loop to check if values wee intered correctly
    if ck <= 1 or ck % 2 != 0:
        print('The list must consist of pairs binary numbers.')
    # For loop to create an ordered pair of binary numbers to add.
    # The solution
    for i in range(1, len(bin_lst)):
        b1 = str(bin_lst[i-1])
        b2 = str(bin_lst[i])
        res, sol = my_bin_adder(b1, b2)
        sol_list.append(sol)
    # return ever other added pair sum. There are actually 6 made.
    # (0+1) (1+2) (2+3) (3+4) (4+5) (5+6)
    return sol_list[::2]


a = "11111"
b = "1010100"
# This function takes two binary numbers and adds them.
# In this case a and b above.
# Results is the steps oder of the addtion used.
results, solution = my_bin_adder(a, b)
print(f'binary numbers {a} + {b} = {solution}')
# Create a dataframe to present the steps used in the addition.
order = pd.DataFrame(results, columns=['Addition'])
order.index += 1  # Starts index at 1
order.index.name = 'Steps'
print(order, '\n')

# Creating a list of binary number to add.
# The a_list must consist of 2 numbers minimum.
a_lst = [11111, 1, 11111, 1010100, 110, 101]
solution2 = bin_add(a_lst)  # call the above function
# convert the list str items to int using builtin function eval().
sol_int = [eval(i) for i in solution2]  # convert the list str items to int
# print(sol_int)  # Print the list.
# Creates the pairs of numbers that were added the [::2] steps every other one.
# Without the [::2] it would display every other one as a pair.
pairs = ([list(zip(a_lst, a_lst[n:])) for n in range(1, len(a_lst))])[0][::2]
# Creates the DataFrame to present the results.
pairs_df = pd.DataFrame(pairs, columns=['binary1 +',
                                        'binary2 = '],
                        index=['case1', 'case2', 'case3'])
pairs_df['solution'] = solution2
print(pairs_df, '\n')

print('EX 5')

print("""What is the effect of allocating more bits to the fraction versus the
characteristic and vice versa?
What is the effect of allocating more bits to the sign?""")

print("""The fraction is where the percision of the calculated number is
obtained the more Bits we use high percision but this could lower the Bits
used by the characteristic lowering the total number size. The sign indicator
is eather positive or negitive thus needs only one Bit 0 (positive) for off
and 1 for on (negitive). Allocating more Bits to the sign is not needed.\n""")

print('EX 6')


def my_ieee_2_dec(ieee):
    ieee = list(ieee)  # create a list from string
    # convert string list to interger list
    sign = int(ieee[0])  # First digit
    exponient = [int(n) for n in ieee[1:12]]  # next 11 digits
    mantissa = [int(x) for x in ieee[12:]]  # the remaining 54 digits
    # Function to remove trailing zeros

    def pop_zeros(items):
        while items[-1] == 0:
            items.pop()

    pop_zeros(mantissa)
    # Calculate exponient e
    exp = 0  # Initialize variables
    sumnum = 0
    bias = 1023  # Bias for 64Bit Double Percision
    # Loop too calculate e exponient starts from end [::-1]
    for i in exponient[::-1]:
        num = (i*2**exp)
        sumnum += num
        exp += 1
    e = sumnum - bias
    # Calculate the fraction
    mexp = 1
    summum = 0
    for i in mantissa:
        mum = (i*2**(-mexp))
        summum += mum
        mexp += 1
    m = summum
    d = ((-1)**sign) * (1 + m) * (2**e)
    return d


ieee1 = '1100000001001000000000000000000000000000000000000000000000000000'
ieee2 = '0100000000001011001100110011001100110011001100110011001100110011'
ieee3 = '0011111111111000010010111011011000010110100011100001110100100000'
floating_decimal1 = my_ieee_2_dec(ieee1)
floating_decimal2 = my_ieee_2_dec(ieee2)
floating_decimal3 = my_ieee_2_dec(ieee3)
print(f'For the 64bit binary number {ieee1} the floating number is '
      f'{floating_decimal1}.')
print(f'For the 64bit binary number {ieee2} the floating number is '
      f'{floating_decimal2}.')
print(f'For the 64bit binary number {ieee3} the floating number is '
      f'{floating_decimal3}.\n')

print('EX 7')


def my_dec_2_ieee(d):
    # getBin is a lambda function that uses bin() to extract binary
    # numbers from the decimal number
    getBin = lambda x: x > 0 and str(bin(x))[2:] or "-" + str(bin(x))[3:]
    # The value acquired from the struct and unstruct functions pack
    # a list of values into a string format 'd' double percision
    # IEEE 754 binary 64 size 8 into a binary h264 code.
    # Then the unpack function 'Q' format creates a value from the
    # binary strings in long long interger size 8.
    val = struct.unpack('Q', struct.pack('d', d))[0]
    # The getBin function converts the decimal interger into
    # binary 64 IEEE 754 string format.
    result = getBin(val)
    # The string format cuts off the 0 at the front of the string. The
    # binary string is filled with zeros on postive numbers to get 64
    # bit length used in the IEEE 754 format.
    if d > 0:
        result = result.zfill(64)
    return result


bin_lst = []
decm_list = [7.5, -309.141740, -25252]
for item in decm_list:
    binstr = my_dec_2_ieee(item)
    print(f'Binary equivalent of {item}: {binstr}\n')
    bin_lst.append(binstr)
bin_dict = dict(zip(decm_list, bin_lst))
binary_df = pd.Series(bin_dict, name='64bit Binary')
pd.set_option('display.colheader_justify', 'center')
bin_df = binary_df.to_frame()
pd.set_option('max_colwidth', 800)
bin_df.index.name = 'decimal'
print(bin_df, '\n')

# To Check answers across all problems with binary numbers.


getBin = lambda x: x > 0 and str(bin(x))[2:] or "-" + str(bin(x))[3:]


def floatToBinary64(value):
    val = struct.unpack('Q', struct.pack('d', value))[0]
    result = getBin(val)
    if value > 0:
        result = result.zfill(64)
    return result


def binaryToFloat(value):
    hx = hex(int(value, 2))
    return struct.unpack("d", struct.pack("q", int(hx, 16)))[0]


# floats are represented by IEEE 754 floating-point format which are
# 64 bits long (not 32 bits!)
# float to binary
num = 126.3
binstr = floatToBinary64(num)
print(f'Binary equivalent of {num}:')
print(binstr + '\n')

# binary to float
fl = binaryToFloat(binstr)
print('Decimal equivalent of ' + binstr)
print(fl, '\n')

print('EX 8')

print("""Define ieee_baby to be a representation of numbers using 6 bits
where the first bit is the sign bit, the second and third bits are allocated
 to the characteristic, and the fourth, fifth, and sixth bits are allocated
to the fraction. The normalization for the characteristic is 1.
Write all the decimal numbers that can be represented by ieee_baby.
 What is the largest/smallest gap in ieee_baby?\n""")
# The equations below comply with the IEEE754 rules for max and min numbers
# for a 6bit system.
largest = (2**(3-1))*((1+sum(0.5**(np.arange(1, 4)))))
smallest = (-1)*(2**(2-1))*(1+0)

print(f"The largest number that can be implimated is bin 011.111 decimal of "
      f"{largest}, and the smallest binary is 111.111 decimal of {smallest}.")
print('The gap is the smallest tolerance between numbers this is easily '
      'calculated from the mantissa in this case which is 2^-3 (1/8) 0.125 '
      'as the numbers get larger the gap widens to 0.5 this gap appears'
      f' in the smallest number caculation {smallest} when the binary '
      'is -2.25 the gap widens to 0.25.\n')


def my_ieee_2_dec(ieee):
    ieee = list(ieee)  # create a list from string
    # convert string list to interger list
    sign = int(ieee[0])  # First digit
    exponient = [int(n) for n in ieee[1:3]]  # next 2 digits
    mantissa = [int(x) for x in ieee[3:]]  # the remaining 3 digits

    # Function to remove trailing zeros not needed for bit size of 6
    def pop_zeros(items):
        while items[-1] == 0:
            items.pop()

    # pop_zeros(mantissa) # not needed for bit size of 6
    # Calculate exponient e
    exp = 0  # Initialize variables
    sumnum = 0
    bias = 1  # Bias
    # Loop too calculate e exponient starts from end [::-1]
    for i in exponient[::-1]:
        num = (i*2**exp)
        sumnum += num
        exp += 1
    e = sumnum - bias
    # Calculate the fraction
    mexp = 1
    summum = 0
    for i in mantissa:
        mum = (i*2**(-mexp))
        summum += mum
        mexp += 1
    m = summum
    d = ((-1)**sign) * (1 + m) * (2**e)
    return d


# Python program to create all
# the possible combinations
def bin_list(bit):
    # generate product in reverse lexicographic order
    bin_lst = [''.join(p) for p in product('10', repeat=bit)]
    # sort by number of ones
    bin_lst.sort(key=lambda s: s.count('1'))
    return bin_lst


bin_list3 = bin_list(6)
# print(com_list3)
# created two new list for dataframe and iterable
bin6 = []
dec6 = []
print(f'The maximum number of values generated is 2^6 or {len(bin_list3)}.')
for i in range(len(bin_list3)):
    bin6.append(bin_list3[i])
    dec = my_ieee_2_dec(bin_list3[i])
    dec6.append(dec)
pd.set_option('display.max_rows', None)
print('Maximum number of binary and decimal numbers that can be constructed '
      'from a 6-bit system')
bin_num_df = pd.DataFrame((bin6), columns=['decimal_no.'],
                          index=np.arange(1, 65))
bin_num_df['binary_no.'] = dec6
print(bin_num_df.sort_values(by=['binary_no.']), '\n')


# 16 bit 'e'
def binary(num):
    return ''.join('{:0>8b}'.format(c) for c in struct.pack('!e', num))


bin16 = binary(2046)
print(bin16, '\n')

print('EX 9\n')
gap_1 = np.spacing(5e15)
print(gap_1)
5e15 == (5e15 + np.spacing(5e15)/3)
print('\n')

print('EX 10')
print('What are some of advantages and disadvantages to binary number'
      ' versus decimal?\n')
print("""The main advantage of using binary is that it is a base which is
 easily represented by electronic devices and for which calculation can be
 carried out using reasonably simple active electronics (simple transistor
circuits), since it only requires on and off (1 and 0) signals (however they
might be represented). Binary data is also reasonable simple to store - again
only needing a two state storage (on/off - 1/0).\n.""")

print("""The main disadvantages are around human usability :
Difficult for most people to read and takes a lot of digits to represent any
 reasonable number (for instance up to 99 million takes 8 digits in Decimal
 and 27 digits in Binary).\n""")

print('EX 11')
print("""The base 1 number for 13 base 10 is 1111111111111, to add additional
 ones 13 = 2 = 15 then it would be 15 1's and to multiply 13 x 2 in base one
 it would be 26 1's up to 64 for 64 bit system.\n""")

print('EX 12')
print("""If you counted in binary on your hands with each finger representing
 a 1 or 0. There would be ten 1's  binary number of 1111111111 for a integer
 number in 64bit system 1023\n""")

print('EX 13')

print("""In binary floating-point arithmetic, division by two can be performed
 by decreasing the exponent by one (as long as the result is not a subnormal
 number) and to multiply by two, all digits shift one place to the left.
 To divide all the digits shift to the right.\n""")
