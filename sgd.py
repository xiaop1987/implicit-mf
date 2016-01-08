import numpy
from sklearn.metrics import mean_squared_error
from math import sqrt
import random

def cal_err(P, Q, R, beta):
	eR = numpy.dot(P,Q)
	e = 0
	for i in xrange(len(R)):
		for j in xrange(len(R[i])):
			if R[i][j] > 0:
				e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)
				for k in xrange(K):
					e = e + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
	return e

def general_gradient_descent(P, Q, R, alpha, beta):
	for i in xrange(len(R)):
		for j in xrange(len(R[i])):
			if R[i][j] > 0:
				eij = R[i][j] - numpy.dot(P[i,:],Q[:,j])
				P[i, :] = P[i, :] + alpha * (2 * eij * Q[:, j].T - beta * P[i, :])
				Q[:, j] = Q[:, j] + alpha * (2 * eij * P[i, :].T - beta * Q[:, j])

class stochastic_gradient_descent:
	def __init__(self, all_indexes):
		self.all_indexes = all_indexes

	def compute(self, P, Q, R, alpha, beta):
		(i, j) = self.all_indexes[random.randint(0, len(self.all_indexes) - 1)]
		eij = R[i][j] - numpy.dot(P[i,:],Q[:,j])
		P[i, :] = P[i, :] + alpha * (2 * eij * Q[:, j].T - beta * P[i, :])
		Q[:, j] = Q[:, j] + alpha * (2 * eij * P[i, :].T - beta * Q[:, j])

def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02, gd_function = general_gradient_descent):
    Q = Q.T
    for step in xrange(steps):
		gd_function(P, Q, R, alpha, beta)
    return P, Q.T

def get_all_nonzero_indexes(R):
	ret = []
	for i in xrange(len(R)):
		for j in xrange(len(R[i])):
			if R[i][j] > 0:
				ret.append((i, j))
	return ret

if __name__ == '__main__':
    R = [
        [5,3,0,1],
        [4,0,0,1],
        [1,1,0,5],
        [1,0,0,4],
        [0,1,5,4] ]

    R = numpy.array(R)

    N = len(R)
    M = len(R[0])
    K = 2
    steps=500000
    alpha=0.0002
    beta=0.000002
    P = numpy.random.rand(N,K)
    Q = numpy.random.rand(M,K)

    nonzero_indexes = get_all_nonzero_indexes(R)
    sgd = stochastic_gradient_descent(nonzero_indexes)
    nP, nQ = matrix_factorization(R, P, Q, K, steps, alpha, beta, sgd.compute)
    print cal_err(nP, nQ.T, R, beta)
    print nP.dot(nQ.T)

