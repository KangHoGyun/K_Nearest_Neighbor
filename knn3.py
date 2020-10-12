import numpy as np
import collections
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
iris = load_iris()

X = iris.data
y = iris.target
y_name = iris.target_names

l =15
for_test = np.array([(i%l == (l-1)) for i in range(y.shape[0])])
for_train = ~for_test

X_train = X[for_train]
y_train = y[for_train]

X_test = X[for_test]
y_test = y[for_test]
# Euclidean Distance Calculator
def Get_Name(ylist):
    if ylist == 0:
        name = y_name[0]
    elif ylist == 1:
        name = y_name[1]
    else:
        name = y_name[2]
    return name

class KNN:
    def __init__(self, x_train, y_train, y_name, k):
        self.x_train = x_train.copy()
        self.y_train = y_train.copy()
        self.y_name = y_name.copy()
        self.k = k

    def Get_Dist(self, a, b):
        return np.linalg.norm(a - b)

    def Majority_Vote(self, xlist, k):
        ylist = []
        self.k = k
        for new_x in xlist:
            distance = [self.Get_Dist(x, new_x) for x in self.x_train]
            d_and_y = zip(distance, self.x_train, self.y_train)
            rank = sorted(d_and_y, key=lambda x: x[0])
            new_y = [v[2] for v in rank[:k]]
            cnt = collections.Counter(new_y)
            y = cnt.most_common(1)[0][0]
            ylist.append(y)
        return np.array(ylist)

    def Weighted_Majority_Vote(self, xlist, k):
        ylist = []
        self.k = k
        for new_x in xlist:
            distance = [self.Get_Dist(x, new_x) for x in self.x_train]
            distance = 1/distance
            d_and_y = zip(distance, self.x_train, self.y_train)
            rank = sorted(d_and_y, key=lambda x: x[0])
            new_y = [v[2] for v in rank[:k]]
            cnt = collections.Counter(new_y)
            y = cnt.most_common(1)[0][0]
            ylist.append(y)
        return np.array(ylist)


knn = KNN(X_train, y_train, y_name, 3)
print ("------------------K == 3----------------------")
print("Majority Vote")
num = 0
for i in range(y_test.shape[0]):
    Ypr = knn.Majority_Vote(X_test, 3)
    print("Test data: ", i, " Computed class: ", Get_Name(Ypr[i]), " True class: ", Get_Name(y_test[i]))
    if Ypr[i] == y_test[i]:
        num = num+1
print ("Concordance Rate: ", (num/y_test.shape[0])*100, "%")

print ("Weighted Majority Vote")
num = 0
for i in range(y_test.shape[0]):
    Ypr = knn.Weighted_Majority_Vote(X_test, 3)
    print("Test data: ", i, " Computed class: ", Get_Name(Ypr[i]), " True class: ", Get_Name(y_test[i]))
    if Ypr[i] == y_test[i]:
        num = num+1
print ("Concordance Rate: ", (num/y_test.shape[0])*100, "%")



