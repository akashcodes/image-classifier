import numpy as np
import math

def sigmoid(varx):
    if type(varx) is list or tuple:
        varx = np.array(varx)
    for i in range(np.shape(varx)[0])
    varx = 1/(1+math.pow(math.e, -varx))
    return varx


def main():
    s = sigmoid([1, 2])
    print(s)

main()