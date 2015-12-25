import sys,os
import time
import numpy as np
import mf

from pyspark import *
import pyspark.mllib.recommendation as recommendation
import test_data_raw

def load_matrix(raw_matrix):
	num_users = len(raw_matrix)
	if num_users < 1:
		print 'Invalid raw_matrix'
		sys.exit(2)

	num_items = len(raw_matrix[0])
	t0 = time.time()
	ret_list = []
	for i in range(num_users):
		for j in range(num_items):
			user, item, count = i, j, raw_matrix[i][j]
			if count != 0:
				ret_list.append((i, j, float(count)))
	t1 = time.time()
	print 'Finished loading matrix in %f seconds' % (t1 - t0)
	return ret_list, num_users, num_items

sconf = SparkConf().setMaster("local")
sc = SparkContext(appName="test")
rank = 10
iterations = 10
lambda_ = 0.01

input_matrix, num_users, num_items = load_matrix(test_data_raw.test_data)
ratings = sc.parallelize(input_matrix)
model = recommendation.ALS.trainImplicit(ratings, rank=4, iterations=iterations, lambda_ = lambda_ , seed=10)
predict_input = ratings.map(lambda x: (x[0], x[1]))
result_local = [[float] * num_items] * num_users
user_factor = model.userFeatures().collect()
product_factor = model.productFeatures().collect()

#for i in range(0, num_users):
#	for j in range(0, num_items):
#		result_local[i][j] = model.predict(i, j)
#		print result_local[i][j]



local_input = mf.load_matrix(test_data_raw.test_data)
local_mf = mf.ImplicitMF(local_input, rank, iterations, lambda_)
local_mf.train_model()
user_factor_mat = np.array(map(lambda x: x[1], user_factor))
product_factor_mat = np.array(map(lambda x:x[1],  product_factor))

print local_mf.predict_matrix()
print user_factor_mat.dot(product_factor_mat.T)
print np.array(test_data_raw.test_data)
#print np.array(result_local)
