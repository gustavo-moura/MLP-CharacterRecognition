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

    def __init__(self, 
        n_input_knots, n_hidden_layers, n_hidden_neurons, 
        n_output_neurons, learning_rate, coefs_hidden, coefs_output):

        self.n_input_knots = n_input_knots
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_neurons = n_hidden_neurons
        self.n_output_neurons = n_output_neurons
        self.learning_rate = learning_rate
        self.coefs_hidden = coefs_hidden
        self.coefs_output = coefs_output


    # Inicia a matriz de coeficientes
    def start_matrix(self):

        # Hidden Layers
        for i in range(self.n_hidden_layers):
            weight_matrix = []

            for k in range(self.n_hidden_neurons):
                weights = []

                for j in range(self.n_input_knots + 1):
                    # +1 correspondente ao weight do bias
                    weights.append(random.random())

                weight_matrix.append(weights)

            self.coefs_hidden.append(weight_matrix)

        # Output Layer
        for k in range(self.n_output_neurons):
            weights = []

            for j in range(self.n_hidden_neurons + 1):
                # +1 correspondente ao weight do bias
                weights.append(random.random())

        self.coefs_output.append(weights)



    def neuron(self, X, W):
        # X são os valores de entrada do neurônio
        # W é a lista de pesos

        v = self.linear_combiner(X, W)

        y = self.activation_function(v)

        return y



    # cap4 - slide 12
    def linear_combiner(self, X, W):
        # x = vetor de entrada (m+1)
        # w = vetor de pesos (m+1)

        # fixed input
        # x[m] = 1
        # w[m] = bias

        # Por paticularidades da linguagem de programação, 
        # ao invés de colocar como x[0], é colocado como x[m].
        # O mesmo vale para w.
        _X = []
        _X += X
        _W = W

        # print(len(_X))
        _X.append(1)
        # print(len(_X))

        if len(_X)!=len(_W):
            print("X[{}] != W[{}]".format(len(_X),len(_W)))
            raise Exception("[BaseMLP][linear_combiner] Input and weight size do not match.\nX[{}] != W[{}]".format(len(X),len(W)))

        v = 0
        for x, w in zip(_X, _W):
            v += float(w)*float(x)

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


    def loss_function(self, real, predicted):
        # Cálculo do Erro Quadrático Médio
        # Least Mean Square Error
        lms = 0
        for d, y in zip(real, predicted):
            lms += (1/2)*((int(d)-y)**2)

        return lms


    def delta_rule(self, r, p):
        # target = real
        # out = predicted

        return -(r-p)+(p*(1-p))


    def forward_propagate(self, X, layer_weights):
        Y = []
        for neuron_weights in layer_weights:
            y_pred = self.neuron(X, neuron_weights)

            Y.append(y_pred)

        return Y


    # Backpropagate error and store in neurons        
    def backward_propagate_error(self, coefs, output, real_class):

        new_coefs

        flag_output_layer = True
        for layer_weights, output_Y in reversed(zip(coefs, output)):

            if flag_output_layer:
                flag_output_layer=False

                for neuron_weights in layer_weights:
                    error = 0.0







                    error = self.loss_function(real_class, output)


                    gradient = self.delta_rule(real, predicted) * predicted_anterior

                    new_weight = old_weight - gradient * self.learning_rate








class MultiLayerPerceptron(BaseMLP):

    # Os métodos implementados para essa classe fazem interface com chamadas de usuários

    def __init__(self, 
        n_input_knots=256, 
        n_hidden_layers=1,  
        n_hidden_neurons=30,
        n_output_neurons=10,
        learning_rate=0.5):

        sup = super(MultiLayerPerceptron, self)

        sup.__init__(
            n_input_knots=n_input_knots, 
            n_hidden_layers=n_hidden_layers,  
            n_hidden_neurons=n_hidden_neurons, 
            n_output_neurons=n_output_neurons,
            learning_rate=learning_rate,
            coefs_hidden=[],
            coefs_output=[])


    def fit(self, X, Y):
        # input layer is X
        # X is list of arrays[256] of 0 and 1 

        # Inicializa a matriz coef
        self.start_matrix()


        # UM epoch
        for input_element, input_class in zip(X, Y):
            
            # Armazenando os valores de output de cada neuronio
            # Cada elemento de output é uma nova lista representando uma camada, 
            # cada elemento dessa sub-lista é a saída de um neurônio 
            output = []
            
            # 1. Propagação da Ativação

            # Camada de Entrada
            # input_element é um vetor[256]
            input_X = input_element


            # Camadas Escondidas
            for layer_weights in self.coefs_hidden:
                output_Y = []

                output_Y = self.forward_propagate(input_X, layer_weights)

                output.append(output_Y)

                input_X = output_Y
                

            # Camada de Saída
            output_Y = []
            output_Y = self.forward_propagate(input_X, self.coefs_output)
            output.append(output_Y)


            # 2. Retropropagação do Erro

            coefs = []
            coefs += self.coefs_hidden
            coefs.append(coefs_output)

            self.backward_propagate_error(coefs, output, input_class)




            # Cálculo do Erro Quadrático Médio
            error = self.loss_function(input_class, output_Y)


            gradient = self.delta_rule(real, predicted) * predicted_anterior

            new_weight = old_weight - gradient * learning_rate


            '''
            prob_pred = max(output_Y)
            class_pred = output_Y.index(prob_pred)


            # Cálculo do Erro
            if class_pred != input_class.index('1'):
                # errou a predição
            else:
                # acertou a predição
            '''

        return self


    def predict(self, X):
        pass













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

    '''

    

