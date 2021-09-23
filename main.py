import numpy as np


class KNearestNeighbour:
    def __init__(self, k):
        self.k = k

    def train(self, x, y):
        self.x_train = x
        self.y_train = y

    def predict(self, x_test):
        distance = self.compute_distance(x_test)
        return self.predict_labels(distance)

    def compute_distance(self, x_test):
        num_test = x_test.shape[0]
        num_train = self.x_train.shape[0]
        distance = np.zeros((num_test, num_train))

        for i in range(num_test):
            for j in range(num_train):
                distance[i, j] = np.sqrt(np.sum((x_test[i, :] - self.x_train[j, :])**2))
        return distance

    def predict_labels(self, distance):
        num_test = distance.shape[0]
        y_predict = np.zeros(num_test)
        for i in range(num_test):
            y_indices = np.argsort(distance[i, :])
            k_closest = self.y_train[y_indices[:self.k]].astype(int)
            y_predict[i] = np.argmax(np.bincount(k_closest))

        return y_predict

if __name__ == '__main__':
    x = np.loadtxt("example_data/data.txt", delimiter=',')
    y = np.loadtxt("example_data/targets.txt")
    k1 = 0
    matches = 0
    for n in range(1, y.shape[0]):
        KNN = KNearestNeighbour(k=n)
        KNN.train(x, y)
        y_predict = KNN.predict(x)
        if sum(y == y_predict) > matches:
            matches = sum(y == y_predict)
            k1 = n
    print(f"Ideal k value : {k1} \nAccuracy : {matches/y.shape[0]*100}%")

