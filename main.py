import random
from NeuralNetworkClassifier import *


def main():
    # Initialized Neural Network with 400 input units
    # 10 output units
    # 1 hidden layer with 25 units
    nn1 = NeuralNetworkClassifier(2, 1, 1, [4])
    training_input = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]
    training_output = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
        [1, 0, 0]
    ]

    nn1.fit(training_input, training_output, 0.03, 70)

    """
    #Initialize neural network
    nn1 = NeuralNetworkClassifier(400, 10, 1, [25])
    
    #Load sample data
    theta1 = np.genfromtxt("theta1.txt")
    theta2 = np.genfromtxt("theta2.txt")
    nn1.thetas[0] = theta1
    nn1.thetas[1] = theta2
    x = np.genfromtxt("x.txt")
    y = np.genfromtxt("y.txt")

    test_x = np.genfromtxt("test_x.txt")
    print("Prediction:", nn1.predict([test_x])+1)
    imgv = np.reshape(test_x, (20, 20))
    plt.imshow(imgv, cmap='gray')
    plt.show(block=False)
    input()
    """
    
    """
    print("Testing data.. q to quit")
    todo = "c"
    while(todo != "q"):
        rn = random.randint(0, 4999)
        #Plot image
        curex = x[rn]
        imgv = np.reshape(curex, (20, 20))
        print("Prediction:", nn1.predict([x[rn]])+1)
        plt.imshow(np.transpose(imgv), cmap='gray')
        plt.show(block=False)
        todo = input()
    """

    """
    p = nn1.predict(x)
    correct = 0
    total = np.shape(y)[0]
    for i in range(total):
        if p[i] == y[i]-1:
            correct += 1
    accuracy = (correct/total)*100
    print(accuracy, "%")
    """

main()