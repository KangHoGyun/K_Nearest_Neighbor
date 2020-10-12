import numpy as np
import matplotlib.pyplot as plt

x1 = np.array([-1, 1, 0, 800, 900, 10000])

def step_function(x):
    return np.array(x>0, dtype=np.int)

def sigmoid(x):
    return (1/(1+np.exp(-x)))

def reLU(x):
    return (np.maximum(x, 0))

x2 = np.array([-1, 4, 7, -9])
def softmax(x):
    exp_a = np.exp(x - np.max(x))
    sum_exp_a = np.sum(exp_a)
    return (exp_a/sum_exp_a)

print(softmax(x1))
print(np.sum(softmax((x1))))

x = np.arange(-5, 5, 0.1)
y = reLU(x)

plt.plot(x, y)
plt.show()
#print(sigmoid(x1))
#x = np.arange(-5, 5, 0.1)
#y = sigmoid(x)

#plt.plot(x, y)
#plt.show()
