import numpy as np
import collections
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
iris = load_iris()

class KNN:
    def __init__(self, x_train, y_train, y_name, k): #X_train은 학습된 iris 데이터, y_train은 iris의 target, y_name은 iris의 이름들, k는 이웃의 숫자이다.
        self.x_train = x_train.copy()
        self.y_train = y_train.copy()
        self.y_name = y_name.copy()
        self.k = k

    def get_dist(self, a, b): # Euclidean Distance 을 구하는 함수이다.
        return np.linalg.norm(a - b) #numpy를 이용하여 구하는 방법이다.

    def majority_vote(self, xdata, k): #majority_vote일 때 구하는 함수. xlist에는 테스트 데이터와 k가 들어갈 것이다.
        ydata = []
        self.k = k #k값이 들어오면 바뀌도록 설정해두었다.
        for new_x in xdata:
            distance = [self.get_dist(x, new_x) for x in self.x_train] #학습된 데이터들과 테스트 데티터 사이의 거리를 구한다.
            temp = zip(distance, self.x_train, self.y_train) #임시로 거리들과 학습된 데이터, 학습된 y의 타겟을 묶는다.
            rank = sorted(temp, key=lambda x: x[0]) #거리를 기준으로 소팅을한다.
            new_y = [v[2] for v in rank[:k]] #이웃의 개수만큼 새롭게 분류한다.
            cnt = collections.Counter(new_y) #새로운 y의 개수를 센다.
            y = cnt.most_common(1)[0][0]  #가장 개수가 많은 것을 리턴하기 위해 most_common method를 사용하였습니다.
            print(y)
            ydata.append(y)
        return np.array(ydata)

    def predict_one(self, new_x, k):
        self.k = k
        distance = [self.get_dist(x, new_x) for x in self.x_train]
        d_and_y = zip(distance, self.x_train, self.y_train)
        rank = sorted(d_and_y, key=lambda x: x[0])
        new_y = [v[2] for v in rank[:self.k]]
        cnt = collections.Counter(new_y)
        #print(cnt.most_common(1))  # value & counter
        y = cnt.most_common(1)[0][0]
        return y

    def weighted_majority_vote(self, xdata, k):#weighted_majority_vote일 때 구하는 함수. 이 함수는 가중치로 1/거리를 이용하였다.
        ydata = []
        weight = list()
        self.k = k
        for new_x in xdata:
            distance = [self.get_dist(x, new_x) for x in self.x_train]
            sum_distance = 0
            for i in distance:
                sum_distance += i # 각 거리들을 더해준다.
            sum_distance = 1/sum_distance #거리들의 합을 역수를 취한다.
            weight = distance * np.array(sum_distance) #그리고 역수값을 각 거리들을 곱하여 weight값을 정한다.
            temp = zip(weight, self.x_train, self.y_train) #이번엔 weight값과 학습된 데이터, 학습된 y의 타겟을 묶는다.
            rank = sorted(temp, key=lambda x: x[0])# weight값을 기준으로 소팅한다.
            new_y = [v[2] for v in rank[:k]]
            cnt = collections.Counter(new_y)
            y = cnt.most_common(1)[0][0]
            ydata.append(y)
        return np.array(ydata)
