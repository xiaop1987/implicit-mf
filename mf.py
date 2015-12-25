import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import time
import sys
import test_data_raw


def load_matrix(raw_matrix):
	num_users = len(raw_matrix)
	if num_users < 1:
		sys.exit(2)

	num_items = len(raw_matrix[0])
	t0 = time.time()
	counts = np.zeros((num_users, num_items))
	total = 0.0
	num_zeros = num_users * num_items
	for i in range(num_users):
		for j in range(num_items):
			user, item, count = i, j, raw_matrix[i][j]
			count = float(count)
			if count != 0:
				counts[user, item] = 1
				total += count
				num_zeros -= 1
	counts = sparse.csr_matrix(counts)
	t1 = time.time()
	print 'Finished loading matrix in %f seconds' % (t1 - t0)
	return counts


class ImplicitMF(object):
	def __init__(self, counts, num_factors=40, num_iterations=30, reg_param=0.8):
		self.counts = counts
		self.num_users = counts.shape[0]
		self.num_items = counts.shape[1]
		self.num_factors = num_factors
		self.num_iterations = num_iterations
		self.reg_param = reg_param

	def train_model(self):
		self.user_vectors = np.random.normal(size=(self.num_users, self.num_factors))
		self.item_vectors = np.random.normal(size=(self.num_items, self.num_factors))

		for i in xrange(self.num_iterations):
			t0 = time.time()
			print 'Solving for user vectors...'
			self.user_vectors = self.iteration(True, sparse.csr_matrix(self.item_vectors))
			print 'Solving for item vectors...'
			self.item_vectors = self.iteration(False, sparse.csr_matrix(self.user_vectors))
			t1 = time.time()
			print 'iteration %i finished in %f seconds' % (i + 1, t1 - t0)

	def iteration(self, user, fixed_vecs):
		num_solve = self.num_users if user else self.num_items
		num_fixed = fixed_vecs.shape[0]
		YTY = fixed_vecs.T.dot(fixed_vecs)
		eye = sparse.eye(num_fixed)
		lambda_eye = self.reg_param * sparse.eye(self.num_factors)
		solve_vecs = np.zeros((num_solve, self.num_factors))

		t = time.time()
		for i in xrange(num_solve):
			if user:
				counts_i = self.counts[i].toarray()
			else:
				counts_i = self.counts[:, i].T.toarray()
			CuI = sparse.diags(counts_i, [0])
			pu = counts_i.copy()
			pu[np.where(pu != 0)] = 1.0
			YTCuIY = fixed_vecs.T.dot(CuI).dot(fixed_vecs)
			YTCupu = fixed_vecs.T.dot(CuI + eye).dot(sparse.csr_matrix(pu).T)
			xu = spsolve(YTY + YTCuIY + lambda_eye, YTCupu)
			solve_vecs[i] = xu
			if i % 1000 == 0:
				print 'Solved %i vecs in %d seconds' % (i, time.time() - t)
				t = time.time()

		return solve_vecs

	def predict_matrix(self):
		return self.user_vectors.dot(self.item_vectors.T)

if __name__ == '__main__':
	input_counts = load_matrix(test_data_raw.test_data)
	imCf = ImplicitMF(input_counts, 100, 200)
	imCf.train_model()
	user_factors = imCf.user_vectors
	item_factors = imCf.item_vectors
	print imCf.predict_matrix()
	pass
