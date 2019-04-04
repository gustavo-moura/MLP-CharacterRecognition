# encoding:utf-8
import re
from itertools import islice
import pandas as pd 

def read_data(digits, labels, filename='digits.data'):

    with open('digits.data', 'r') as file_opened:
        file = file_opened.read()

    # digits = []
    # labels = []

    digit = []

    pixels = []
    pixel = ''
    count = 0

    is_digit = True
    label = ''

    file = iter(file)
    for n in file:

        if is_digit:
            if n==' ' or n=='\n':
                pixels.append(pixel)
                pixel=''
                count += 1
            else:
                pixel += n
                next(islice(file, 4, 5), '')


            if count == 256:
                digit.append(pixels)
                pixels = []
                count = 0
                is_digit = False    

        else:        
            if count == 20:
                label = re.sub(' ', '', label)
                count = 0
                is_digit = True
            else:     
                label += n
                count += 1

        if n == '\n':
            digits.append(digit)
            digit = []

            labels.append(label)
            label = ''
    

def show_digit(data, height=16, width=16):
    i = 0
    for h in range(height):
        for w in range(width):
            if data[i]=='1' or data[i]==1:
                print('#', end='')
            else:
                print('.', end='')
            i+=1
        print()


def show_label(label):
    i = label.index('1')
    print('label: ', i)
    return i



digits, labels = [],[]
read_data(digits, labels)

for i in range(len(digits)):
    show_digit(digits[i][0])
    show_label(labels[i])
