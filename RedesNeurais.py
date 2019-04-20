#!/usr/bin/env python
# coding: utf-8

import random
import math

'''
https://scikit-learn.org/stable/modules/neural_networks_supervised.html
https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier
https://www.ibm.com/support/knowledgecenter/en/SSLVMB_22.0.0/com.ibm.spss.statistics.algorithms/alg_mlp_architecture_activation-functions.htm
https://www.researchgate.net/figure/Multilayer-perceptron-artificial-neural-network-a-ANNs-Architecture-b-Phases-of_fig1_307845736
https://hackernoon.com/how-to-initialize-weights-in-a-neural-net-so-it-performs-well-3e9302d4490f
http://deeplearning.net/tutorial/mlp.html
https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
https://www.google.com/search?q=fun%C3%A7%C3%A3o+de+ativa%C3%A7%C3%A3o&source=lnms&tbm=isch&sa=X&ved=0ahUKEwiLlMOKy9jhAhVDIrkGHcjSA70Q_AUIDygC&biw=1366&bih=657#imgrc=luwk7aPFWfVJMM:
'''

class BaseMLP:

    def __init__(self, n_input_knots, n_hidden_layers, n_hidden_neurons, n_output_neurons, coefs_hidden, coefs_output, bias):

        self.n_input_knots = n_input_knots
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_neurons = n_hidden_neurons
        self.n_output_neurons = n_output_neurons
        self.coefs_hidden = coefs_hidden
        self.coefs_output = coefs_output
        self.bias = bias


    # Inicia a matriz de coeficientes
    def start_matrix(self):

        # Hidden Layers
        for i in range(self.n_hidden_layers):
            weight_matrix = []

            for k in range(self.n_hidden_neurons):
                weights = []

                for j in range(self.n_input_knots):
                    weights.append(random.random())

                weight_matrix.append(weights)

            self.coefs_hidden.append(weight_matrix)

        # Output Layer
        for k in range(self.n_output_neurons):
            weights = []

            for j in range(self.n_output_neurons):
                weights.append(random.random())

            self.coefs_output.append(weights)





    def neuron(self, X, W):

        v = self.linear_combiner(X, W)

        y = self.activation_function(v)


        return y



    # cap4 - slide 12
    def linear_combiner(self, x, w):
        # x = vetor de entrada (m+1)
        # w = vetor de pesos

        # fixed input
        # x[m] = 1
        # w[m] = bias

        # Por praticidades da linguagem de programação, ao invés de colocar como x[0], é colocado como x[m]
        # x.append(1)
        # w.append(bias)

        # Por praticidade é inicializada a somatória da saída já com o valor do bias, sendo equivalente ao mencionado anteriormente

        v = self.bias
        for i in range(self.n_hidden_neurons):
            v += w[i]*x[i]

        return v



    def activation_function(self, v):
        # Linear
        # return v

        # Degrau
        # if v>0:
        #   return 1
        # else:
        #   return 0

        # Sigmóide - Logística
        return (1/(1+math.exp(-v)))


class MultiLayerPerceptron(BaseMLP):



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

    '''
    n_input_knots = 256

    n_hidden_layers = # input

    n_output_neurons = 10

    n_hidden_neurons = 10

    coefs_hidden = []
    o iésimo elemento da lista representa a matriz de pesos correspondentes a camada i

    coefs_output

    bias = #?
    '''

    def __init__(self, 
        n_input_knots=256, 
        n_hidden_layers=100,  
        n_hidden_neurons=256,
        n_output_neurons=10, 
        bias=0.1):

        sup = super(MultiLayerPerceptron, self)

        sup.__init__(
            n_input_knots=n_input_knots, 
            n_hidden_layers=n_hidden_layers,  
            n_hidden_neurons=n_hidden_neurons, 
            n_output_neurons=n_output_neurons,
            coefs_hidden=[],
            coefs_output=[],
            bias=bias)


    def fit(self, X, Y):
        # input layer is X
        # X is list of arrays[256] of 0 and 1 

        # Inicializa a matriz coef
        self.start_matrix()


        # UM epoch
        for input_element, input_class in zip(X, Y):
            
            
            # 1. propagação da ativação

            # camada de entrada
            input_X = input_element
            # input_element é um vetor[256]

            # camadas escondidas
            for layer_weights in self.coefs_hidden:
                output_Y = []

                for neuron_weights in layer_weights:
                    y_pred = self.neuron(input_X, neuron_weights)

                    output_Y.append(y_pred)

                input_X = output_Y


            # camada de saída
            output_Y = []
            for neuron_weights in coefs_output:
                y_pred = self.neuron(input_X, neuron_weights)

                output_Y.append(y_pred)


            # cálculo do erro quadrático médio
            lsm = 0
            for d, y in zip(input_class, output_Y):
                lms += ((d+y)**2)/2

            
           
            '''
            prob_pred = max(output_Y)
            class_pred = output_Y.index(prob_pred)


            # calculo do erro
            if class_pred != input_class.index('1'):
                # errou a predição
            else:
                # acertou a predição
            '''

            # 2. retropropagação do erro



    def predict(self, X):
        pass



    

