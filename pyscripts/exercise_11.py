#!/usr/bin/env python
# coding: utf-8

import csv
import numpy as np
import pickle
import json

print('EX1, EX2, and EX3\n'
      'Write a list so that each item can be put on one'
      ' line and write to text file one line at a time.\n Save the same list'
      ' to a csv file.\n')

data = [['Ford', 'Chev', 'Dodge', 'Toyota'],
        ['F250', '2500', 'Ram', 'TRD_Tundra'],
        [3.8, 5.6, 6.2, 5.7],
        [67000, 72000, 54000, 58000]]

print(f'Write txt file to disk with 2D list\n {data}\n')
with open('ex1.txt', 'w') as f:
    f.write("\n".join(str(item) for item in data))
    # The code below will produce the same as the above code.
    # for item in data:
    # f.write("%s\n" % str(item))
f.close()
print(f'Write csv file to disk with 2D list\n {data}\n')
# On windows the newline='' must be in the open statement
# behine the 'w' staement or a blank line will separte each
# item in the list. it by default will it will write \r\r\n
# on Windows, where the default text mode will translate
# each \n into \r\n (enter a new line).
with open('ex1.csv', 'w', newline='') as fcsv:
    csvwriter = csv.writer(fcsv)
    csvwriter.writerow(data[0])
    csvwriter.writerows(data[1:4])
print('Open csv file with the data list stored and append list'
      'with additional data. Appended data list.')
with open('ex1.csv', 'r') as fcsv:
    cars = list(csv.reader(fcsv))
cars = list(cars)
cars.append(['v6', 'v8', 'v8', 'v6'])
print(cars, '\n')

arr_data = np.array(data)
print(f'Create a np array and write it to disk as csv\n {arr_data}\n'
      ' and load the data.')
np.savetxt('arr_data.csv', arr_data,
           delimiter=',', fmt='%s', header='Car1, Car2, Car3, Car4')
my_cars = np.loadtxt('arr_data.csv', delimiter=',', dtype=str)
print(my_cars, '\n')
pickle.dump(data, open('data.pkl', 'wb'))
pickle_data = pickle.load(open('data.pkl', 'rb'))
print(f'EX4\nPickle data file save and load\n {pickle_data}\n')

print('EX5\n\t Write and Read a JSON File')
phone_dict = {'name': ['Russell Smith', 'Kathy Mata',
                       'Oscar Lopez', 'Lorena Lopez'],
              'phone_no': ['832-360-7274', '713-539-7909',
                           '832-391-1988', '832-391-1989'],
              'age': ['63', '63', '18', '15'],
              'occupation': ['IT', 'Eng', 'student', 'student']}
print(f'Created dictonary to dump to a json file.\n{phone_dict}\n')
json.dump(phone_dict, open('phone_dict.json', 'w'))
phone_lst = json.load(open('phone_dict.json', 'r'))
print(f'The JSON file loaded from disk\n {phone_lst}\n')

print('EX6')
arrdata = np.arange(1.0, 10.0)
# json can not handle an numpy array, so it was converted to a list.
arrdata = arrdata.tolist()
Jdata = {'data': (arrdata)}  # created a json variable dict w/list
print(f'Created numpy array to dump to a json file.\n{Jdata}\n')
json.dump(Jdata, open('Jdata.json', 'w'))
Data_Json = json.load(open('Jdata.json', 'r'))
# Converted the data back to np.array.
Data_Json = np.array(Data_Json)
print(f'The JSON file loaded from disk\n {Data_Json}\n')
