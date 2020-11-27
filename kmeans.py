import numpy as np
import numpy.random as random

class K_Means:
	def __init__(self, k=2, max_iter=100):
		self.k = k
		self.max_iter = max_iter

	def fit(self, data):
		self.centers = {}
		for i in range(self.k):
			self.centers[i] = data[i]

		for i in range(self.max_iter):
			self.clf = {}
			for j in range(self.k):
				self.clf[j] = []
			for feature in data:
				distances = []
				for center in self.centers:
					distances.append(np.linalg.norm(feature - self.centers[center]))
				classification = distances.index(min(distances))
				self.clf[classification].append(feature)

			for c in self.clf:
				self.centers[c] = np.average(self.clf[c], axis=0)

	def predict(self, p_data):
		distances = [np.linalg.norm(p_data - self.centers[center]) for center in self.centers]
		index = distances.index(min(distances))
		return index

if __name__ == '__main__':
	x = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
	k_means = K_Means(k=3)
	k_means.fit(x)
	p_data = random.randint(10, size=2)
	print(p_data)
	pred_res = k_means.predict(p_data)
	print(pred_res)