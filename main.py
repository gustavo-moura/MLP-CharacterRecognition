#!/usr/bin/env python
# coding: utf-8

import re
import pandas as pd 
from itertools import islice

from RedesNeurais import MultiLayerPerceptron


# Função utilizada para abrir um aquivo, ler e retornar as informações de digits e labels
def read_data(digits, labels, filename='digits.data'):

    with open('digits.data', 'r') as file_opened:
        file = file_opened.read()

    # digits = []
    # labels = []


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
                digit = pixels
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
    

# Interpreta e exibe no terminal os dígitos, somente para facilitar a visualização
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


# Interpreta o valor de label
def show_label(label):
    i = label.index('1')
    print('label: ', i)
    return i



# MAIN

digits, labels = [], []
read_data(digits, labels)

i = 0
show_digit(digits[i])
show_label(labels[i])

mlp = MultiLayerPerceptron()




