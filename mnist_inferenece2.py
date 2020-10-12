import numpy as np
import sys, os
import pickle

sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from sngtae import sigmoid, softmax


class MNIST_Predict():
    def __init__(self):
        self.x_test = []
        self.t_test = []

        self.W1 = []
        self.W2 = []
        self.W3 = []
        self.b1 = []
        self.b2 = []
        self.b3 = []

    def load_data(self):
        (x_train, t_train), (self.x_test, self.t_test) = \
            load_mnist(normalize = True, flatten = True, one_hot_label = False)

    def init_network(self):
        with open("sample_weight.pkl", 'rb') as f:
            network = pickle.load(f)
        self.W1 = network['W1']
        self.W2 = network['W2']
        self.W3 = network['W3']
        self.b1 = network['b1']
        self.b2 = network['b2']
        self.b3 = network['b3']
        return network

    def predict(self, x):
        a1 = np.dot(x, self.W1) + self.b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, self.W2) + self.b2
        z2 = sigmoid(a2)
        a3 = np.dot(z2, self.W3) + self.b3
        y = softmax(a3)

        return y

mnist = MNIST_Predict()
mnist.load_data()
mnist.init_network()

num = 5
count = 0
for i in range(num):
    y = mnist.predict(mnist.x_test[i])
    print(y)
    print (np.argmax(y))