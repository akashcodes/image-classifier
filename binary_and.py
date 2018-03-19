import random
from NeuralNetworkClassifier import *
import numpy as np


def main():

    #Initialize neural network
    nn1 = NeuralNetworkClassifier(2, 1, 1, [5])
    nn1.thetas[0] = np.random.rand(4, 3)
    nn1.thetas[1] = np.random.rand(2, 5)

    p = nn1.predict([[0, 1]])

    print(p)


main()