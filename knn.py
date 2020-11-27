
from collections import Counter

import numpy as np
import numpy.random as random

class KNN:

	def __init__(self, k):
		self.k = k
		self.X_train = None
		self.y_train = None

	def fit(self, X_train, y_train):
		self.X_train = X_train
		self.y_train = y_train

	def predict(self, X_predict):
		distances = [np.linalg.norm(X_predict - x_train) for x_train in self.X_train]
		nearest = np.argsort(distances)
		topK_y = [self.y_train[i] for i in nearest[:self.k]]
		votes = Counter(topK_y)
		y_predict = votes.most_common(1)[0][0]
		return y_predict

if __name__ == '__main__':
	X_train = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
	y_train = np.array([1, 1, 2, 2, 1, 2])
	knn = KNN(3)
	knn.fit(X_train, y_train)
	p_data = random.randint(10, size=2)
	print(p_data)
	pred_res = knn.predict(p_data)

	print(pred_res)