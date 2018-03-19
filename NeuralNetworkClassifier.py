import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit as sigmoid

def derivative_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

class NeuralNetworkClassifier():
    
    def __init__(self, input_size, output_size, nlayers, layer_sizes):
        self.thetas = []
        self.input_size = input_size
        layer_sizes.insert(0, input_size)
        layer_sizes.append(output_size)
        for i in range(nlayers+1):
            self.thetas.append(np.zeros([layer_sizes[i+1], layer_sizes[i]+1]))
        print(self.thetas)
        print("Neural Network Initialized!")
    
    
    def fit(self, inputs, outputs, alpha, reg_lambda):
        #Verification of Input
        if type(inputs) is list or tuple:
            inputs = np.array(inputs)
        dim = np.shape(inputs)
        if dim[1] != self.input_size:
            print("Number of features is different from the training examples:\nGiven =>", dim[1], "\nRequired =>", self.input_size)
            return
        
        predictions = self.predict(inputs)
        m = len(predictions)

        for i in range(m):
            deltas = []
            error = predictions[i] - outputs[i]
            error = error[:, np.newaxis]
            deltas.append(error)
            # From theta last to theta fisrt
            for j in range(len(self.thetas)-1, -1, -1):
                delta = np.dot(deltas[0], self.thetas[j])
                deltas.insert(0, delta)
        return deltas
    

    def backward_propagate(self):
        return


    def predict(self, inputs):
        #Verification of Input
        if type(inputs) is list or tuple:
            inputs = np.array(inputs)
        dim = np.shape(inputs)
        if dim[1] != self.input_size:
            print("Number of features is different from the training examples:\nGiven =>", dim[1], "\nRequired =>", self.input_size)
            return
        
        #Feedforward
        z = inputs
        for i in self.thetas:
            a = np.insert(z, 0, 1, axis=1)
            print(a)
            a = np.dot(a, np.transpose(i))
            z = sigmoid(a)
        
        #Predictions Done
        print(z)
        return z
