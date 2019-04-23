#!/usr/bin/env python
# coding: utf-8

from RedesNeurais import MultiLayerPerceptron
from handler import read_data, show_digit, show_label, train_test_split, accuracy


# Leitura do arquivo
digits, labels = [], []
read_data(digits, labels, filename="digits.data")


# Exibir alguns itens para conferência
# 	TODO: comentar esse trecho
i = 0
# show_digit(digits[i])
# show_label(labels[i])


# TODO: fazer for pra k-fold cross validation

# Separar conjunto de treinamento e teste
x_train, x_test, y_train, y_real = train_test_split(digits, labels, test_size=0.2, shuffle=True) #, stratify=y)


# Criar rede neural
mlp = MultiLayerPerceptron().fit(x_train, y_train)

y_pred = mlp.predict(x_test)

# print(accuracy(y_real, y_pred))




