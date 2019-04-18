import re
from itertools import islice


# Função utilizada para abrir um aquivo, ler e retornar as informações de digits e labels
def read_data(digits, labels, filename='digits.data'):

    with open('digits.data', 'r') as file_opened:
        file = file_opened.read()

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
    


# Separa os dados e labels em conjuntos de treinamento e teste
def train_test_split(X, Y, test_size=0.2, shuffle=True):

    if shuffle:
        c = list(zip(X, Y))
        random.shuffle(c)
        _x, _y = zip(*c)
    else:
        _x = X
        _y = Y


    index = int(len(_y)*test_size)

    x_test = _x[:index]
    y_test = _y[:index]

    x_train = _x[index:]
    y_train = _y[index:]


    return x_train, x_test, y_train, y_test