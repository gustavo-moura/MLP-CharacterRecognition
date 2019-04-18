#!/usr/bin/env python
# coding: utf-8

import random

class MultiLayerPerceptron:

	'''
	Atributes
	classes_ : array or list of array of shape (n_classes,)
	Class labels for each output.

	loss_ : float
	The current loss computed with the loss function.

	coefs_ : list, length n_layers - 1
	The ith element in the list represents the weight matrix corresponding to layer i.

	intercepts_ : list, length n_layers - 1
	The ith element in the list represents the bias vector corresponding to layer i + 1.

	n_iter_ : int,
	The number of iterations the solver has ran.

	n_layers_ : int
	Number of layers.

	n_outputs_ : int
	Number of outputs.

	out_activation_ : string
	Name of the output activation function.
	'''
	n_input_knots = 256

	n_hidden_layers = input

	n_output_layers = 10

	n_neurons = 10

	out_activation: string
	Name of the output activation function.

	coefs = []
	o iésimo elemento da lista representa a matriz de pesos correspondentes a camada i



	def __init__(self, n_hidden_layers):
		self.n_hidden_layers = n_hidden_layers
		self.coefs = [] # n_layers - 1


	def fit(self, X, y):
		# input layer is X
		# X is an array[256] of 0 and 1 


		- propagação da ativação

		- retropropagação do erro



	def predict(self, X):
		pass






	# Inicia a matriz de coeficientes
	def start_matrix():
		for i in range(self.n_layers):
			weight_matrix = []

			for k in range(n_neurons):
				weights = []

				for j in range(n_input_knots):
					weight.append(random())

				weight_matrix.append(weights)

		coefs.append(weight_matrix)




	# cap4 - slide 12
	def neuron(x):
		# x = vetor de entrada (m+1)

		# fixed input
		# x[m] = 1
		# Por praticidades da linguagem de programação, ao invés de colocar como x[0], é colocado como x[m]

		# w[m] = bias

		v = 0
		for i in range(m):
			v += w[i]*x[i]


		v = hard_limiter(v)



		return v



	# TODO: terminar de escrever e conferir funções de ativação
	def hard_limiter(v):
		# Linear
		return v

		# Degrau
		if v>0:
			return 1
		else:
			return 0

		# Logística







