#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 07:30:32 2022

@author: rlsmi
"""
import matplotlib.pyplot as plt
import math

print('EX 1')
print("""Describe the differences between classes and objects.""")
print("""A class is a blueprint used to define a logical grouping of data and
functions, while an object is the instance of the defined class with the
actual values.\n""")

print("EX 2")
print("Describe why we use self as the first argument in a method.")
print("""The class instance method must have and extra argument to as the
first argument (self) which refers back to the object itself. Instance methods
can freely access attributes and other methods in the same object by using
 self.\n""")

print("EX 3")
print('What ia constructor and why do we use it?')
print("""Constructors are generally used for instantiating an object. The task
of constructors is to initialize(assign values) to the data members of the
class when an object of the class is created. In Python the __init__() method
is called the constructor and is always called when an object is created.\n""")

print("EX 4")
print("""Describe the differences between classes and instance attributes.""")
print("""A class is a blueprint used to define a logical grouping of data and
functions. Defines a grouping of objects. Classes are created by a
key word class Attributes are the variables that belong to a class.
Attributes are always public and can be accessed using the dot (.) operator.
State: It is represented by the attributes of an object. It also reflects the
properties of an object. Behavior: It is represented by the methods of an
object. It also reflects the response of an object to other objects.
Identity: It gives a unique name to an object and enables one object to
interact with other objects. Instance attributes are those tha belong to a
specific instance and to use them you must use self.attribute within the
class (belonging to the class). Class attributes can be used anywhere within
the class without self.\n""")

print("EX 5 and 6")


class Point():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def plot_points(self):
        plt.scatter(self.x, self.y, marker='o', color='blue')
        plt.plot(self.x, self.y, 'bo', linestyle="--")
        plt.show()

    def distance(self):
        if self.x == list and self.y == list:
            d = ((self.y[0]-self.x[0])**2 + (self.y[1] - self.x[1])**2)**.5
        else:
            d = math.dist(self.x, self.y)

        print(f'The distance between points {self.x} and {self.y} is {d}\n')


loc = Point((2, 8), (10, 14))
loc.plot_points()
loc.distance()

print("EX 7")
print("""What is inheritance?""")
print("""Inheritance enables us to define a class that takes all the
functionality from a parent class and allows us to add more. It refers to
defining a new class with little or no modification to an existing class.
The new class is called derived(or child) class and the one from which it
inherits is called the base (or parent) class.\n""")

print("EX 8")
print("""How do we inherit from a superclass and add a new method?""")
print("""This is done by writing a new class that references
(inherits from parent) the parent class. class Newclass(Parent): We
can then write a new method (function) with in the new class.
def new_method(self):. The child class will inherit all the attributes and
 methods.\n""")

print("EX 9")
print("""When we inherit from a superclass, we need to replace a methods with
a new one; how do we do that?""")
print("""The class from which a class inherits is called the parent or
superclass. A class which inherits from a superclass is called a subclass,
also called heir class or child class. Superclasses are sometimes called
ancestors as well. There exists a hierarchical relationship between classes.
This is done by writing function with added attributes and accessing the
parent __init__function. This will allow you to change the attributes and
still maintain all the attributes from parent.\n""")

print("EX 10")
print("""Whats the super method and why do we need it?""")
print("""The super() function is used to give access to methods
and properties of a parent or sibling class. The super() function
returns an object that represents the parent class. It allow the
code to be maintainable for the foreseeable future.\n""")

print("EX 11")


class Operator():
    n = 0

    def __init__(self, a, b):
        self.a = float(a)
        self.b = float(b)
        Operator.n += 1

    def addition(self):
        print(f'{self.a} + {self.b} = {self.a + self.b}')

    def subtraction(self):
        print(f'{self.a} - {self.b} = {self.a - self.b}')

    def multiplication(self):
        print(f'{self.a} x {self.b} = {self.a * self.b}')

    def division(self):
        print(f'{self.a} / {self.b} = {self.a / self.b}')

    def square(self):
        print(f'{self.a}^2 and {self.b}^2 = {(self.a)**2} , {(self.b)**2}')


class Power(Operator):

    def __init__(self, a, b, x, y):
        super().__init__(a, b)
        self.x = float(x)
        self.y = float(y)

    def power(self):
        print(f'{self.y}^{self.x} = {self.y**self.x}')

    def dist(self):
        self.__distance = round(((self.x-self.a)**2
                                 + (self.y-self.b)**2)**.5, 3)
        print(f'The distance between points {self.a}, {self.b} and'
              f' {self.x}, {self.y} ='
              f' {((self.x-self.a)**2 + (self.y-self.b)**2)**.5}')

    def get_distance(self):
        print(f'The distance is : {self.__distance}')

    def Num_inits(self):
        print(f'The number of instances used for calculations = {Operator.n}')


op = Operator(24, 56)
pw = Power(24, 12, 10, 3)
op.addition()
op.subtraction()
op.multiplication()
op.division()
op.square()
print('\nUsing inheriting and updating attributes with super()'
      ' including encapsulation')
pw.power()
pw.dist()
pw.get_distance()
pw.Num_inits()
print('\n')
