#https://stackabuse.com/creating-a-neural-network-from-scratch-in-python-adding-hidden-layers/
import math


def sigmoid(v):
	return (1/(1+math.exp(-v)))






def backward_propagate():


# Phase 1
	

dzo_dwo = ah

dao_dzo = sigmoid(zo) * (1 - sigmoid(zo))       #6

dcost_dao = (ao-labels)                         #5


dcost_dwo = dcost_dao * dao_dzo * dzo_dwo          #1


new_weight = current_weight - learning_rate * dcost_dwo



# Phase 2

dzh_dwh = input_features                        #9

dah_dzh = sigmoid(zh) * (1 - sigmoid(zh))       #8

dzo_dah = wo                                 #7 
dcost_dzo = dcost_dao * dao_dzo              #4 
dcost_dah = dcost_dzo * dzo_dah                 #3

dcost_dwh = dcost_dah * dah_dzh * dzh_dwh          #2


new_weight = current_weight - learning_rate * dcost_dwh




