import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

from knn import KNN

iris = load_iris()

X = iris.data
y = iris.target
y_name = iris.target_names
print(type(X))
print(y_name)
l =15
for_test = np.array([(i%l == (l-1)) for i in range(y.shape[0])])
for_train = ~for_test

X_train = X[for_train]
y_train = y[for_train]

X_test = X[for_test]
y_test = y[for_test]

def get_name(ylist): #해당 꽃의 이름을 target을 보고 분류하는 함수이다.
    if ylist == 0:
        name = y_name[0]
    elif ylist == 1:
        name = y_name[1]
    else:
        name = y_name[2]
    return name

knn = KNN(X_train, y_train, y_name, 3)
print ("----------------------------------K == 3--------------------------------------")
print("Majority Vote")# Majority Vote인 경우
num = 0
for i in range(y_test.shape[0]):
    Ypr = knn.majority_vote(X_test, 3)
    print("Test data: ", i, " Computed class: ", "{:10}".format(get_name(Ypr[i])), " True class: ", get_name(y_test[i]))
    if Ypr[i] == y_test[i]:
        num = num+1
print ("Concordance Rate: ", (num/y_test.shape[0])*100, "%") #정확도 계산

print ("Weighted Majority Vote")#Weighted Majority Vote인 경우
num = 0
for i in range(y_test.shape[0]):
    Ypr = knn.weighted_majority_vote(X_test, 3)
    print("Test data: ", i, " Computed class: ", "{:10}".format(get_name(Ypr[i])), " True class: ", get_name(y_test[i]))
    if Ypr[i] == y_test[i]:
        num = num+1
print ("Concordance Rate: ", (num/y_test.shape[0])*100, "%")


print ("----------------------------------K == 5--------------------------------------")
print("Majority Vote")
num = 0
for i in range(y_test.shape[0]):
    Ypr = knn.majority_vote(X_test, 5)
    print("Test data: ", i, " Computed class: ", "{:10}".format(get_name(Ypr[i])), " True class: ", get_name(y_test[i]))
    if Ypr[i] == y_test[i]:
        num = num+1
print ("Concordance Rate: ", (num/y_test.shape[0])*100, "%")

print ("Weighted Majority Vote")
num = 0
for i in range(y_test.shape[0]):
    Ypr = knn.weighted_majority_vote(X_test, 5)
    print("Test data: ", i, " Computed class: ", "{:10}".format(get_name(Ypr[i])), " True class: ", get_name(y_test[i]))
    if Ypr[i] == y_test[i]:
        num = num+1
print ("Concordance Rate: ", (num/y_test.shape[0])*100, "%")

print ("----------------------------------K == 10--------------------------------------")
print("Majority Vote")
num = 0
for i in range(y_test.shape[0]):
    Ypr = knn.majority_vote(X_test, 10)
    print("Test data: ", i, " Computed class: ", "{:10}".format(get_name(Ypr[i])), " True class: ", get_name(y_test[i]))
    if Ypr[i] == y_test[i]:
        num = num+1
print ("Concordance Rate: ", (num/y_test.shape[0])*100, "%")

print ("Weighted Majority Vote")
num = 0
for i in range(y_test.shape[0]):
    Ypr = knn.weighted_majority_vote(X_test, 10)
    print("Test data: ", i, " Computed class: ", "{:10}".format(get_name(Ypr[i])), " True class: ", get_name(y_test[i]))
    if Ypr[i] == y_test[i]:
        num = num+1
print ("Concordance Rate: ", (num/y_test.shape[0])*100, "%")