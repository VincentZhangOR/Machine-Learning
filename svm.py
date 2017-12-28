#!/usr/bin/env python

from __future__ import division
from sklearn import svm
import numpy as np
# from matplotlib import pyplot
from collections import defaultdict
import time
import sys
import math

def create_feature_map(train_file):
	column_values = defaultdict(set)
	for line in open(train_file):
		line = line.strip()
		features = line.split(", ")[:-1] # last field is target
		for i, fv in enumerate(features):
			column_values[i].add(fv)
	feature2index = {(-1, 0): 0} # bias
	for i, values in column_values.iteritems():
		for v in values:            
			feature2index[i, v] = len(feature2index)
	dimension = len(feature2index)
	print "Dimensionality: ", dimension
	return feature2index

def map_data(filename, feature2index):
	X = []
	Y = []
	# dimension = len(feature2index)
	for j, line in enumerate(open(filename)):
		line = line.strip(',\n')
		features = line.split(", ")
		feat_vec = np.zeros(dimension)
		if filename == 'income.test.txt':
			for i, fv in enumerate(features):
				if (i, fv) in feature2index: # ignore unobserved features
					feat_vec[feature2index[i, fv]] = 1
			feat_vec[0] = 1 # bias
			X.append(feat_vec)
		else:
			for i, fv in enumerate(features[:-1]): # last one is target
				if (i, fv) in feature2index: # ignore unobserved features
					feat_vec[feature2index[i, fv]] = 1
			feat_vec[0] = 1 # bias
			X.append(feat_vec)
			Y.append(1 if features[-1] == ">50K" else -1)
	return (X,Y)

def train(train_data, dev_data, C):
	train_X, train_Y = train_data
	clf = svm.SVC(kernel = 'linear', C=C)
	start = time.time()
	clf.fit(train_X,train_Y)
	end = time.time()
	print '-----1.1 & 1.2-----'
	print 'C:', C
	print 'Training time:', round(end - start,2), 's'
	print 'Training-set error rate:', '{:.2%}'.format(1 - clf.score(train_X,train_Y))
	dev_X, dev_Y = dev_data
	print 'Dev-set error rate:', '{:.2%}'.format(1 - clf.score(dev_X,dev_Y))
	print 'Number of support vectors:', sum(clf.n_support_)
	support_vectors = clf.support_vectors_
	return ((clf.coef_[0], clf.intercept_[0]), support_vectors)

def calculation(w, train_data, dev_data, C):
	total_slacks = 0
	train_X, train_Y = train_data
	train_error_count = 0
	for (x,y) in zip(train_X, train_Y):
		temp = y * np.dot(w, x)
		total_slacks += max(0, 1 - temp)
		if temp < 0:
			train_error_count += 1

	total_objective = 1/2 * np.dot(w,w) + C * total_slacks
	train_error_rate = train_error_count / len(train_Y)
	
	dev_X, dev_Y = dev_data
	dev_error_count = 0
	for (x,y) in zip(dev_X, dev_Y):
		temp = y * np.dot(w, x)
		if temp < 0:
			dev_error_count += 1
	dev_error_rate = dev_error_count / len(dev_Y)

	return (total_objective, train_error_rate, dev_error_rate)


def pegasos(train_data, dev_data, T, C):
	train_X, train_Y = train_data
	Lambda = 2 / (len(train_Y) * C)
	w = np.array([0.0 for x in xrange(len(train_X[0]))])
	T =  Epoch * len(train_Y)
	i = 0
	epoch = 0
	start = time.time()
	for t in xrange(1, T+1):
		x, y = train_X[i], train_Y[i]
		eta = 1 / (Lambda * t)
		if y * np.dot(w, x) < 1:
			w = (1-eta*Lambda) * w + eta * y * x
		else:
			w = (1-eta*Lambda) * w
		# w = min(1, (1/math.sqrt(Lambda)/math.sqrt(np.dot(w,w)))) * w
		i += 1
		if i == len(train_Y):
			i = 0
			epoch += 1
			# (objective, train_error_rate, dev_error_rate) = calculation(w, train_data, dev_data, C)
			# print 'Epoch:', epoch, 'Objective:', round(objective,2), 'Train error:', '{:.2%}'.format(train_error_rate), 'Dev error:', '{:.2%}'.format(dev_error_rate)
	end = time.time()
	print 'Pegasos training time:', round(end - start,2), 's'
	return w

def quad_train(train_data, dev_data, test_data, C):
	train_X, train_Y = train_data
	clf = svm.SVC(kernel = 'poly', degree = 2, coef0 = 1, C=C)
	start = time.time()
	clf.fit(train_X,train_Y)
	end = time.time()
	print 'C:', C
	print 'Training time:', round(end - start,2), 's'
	print 'Training-set error rate:', '{:.2%}'.format(1 - clf.score(train_X,train_Y))
	dev_X, dev_Y = dev_data
	print 'Dev-set error rate:', '{:.2%}'.format(1 - clf.score(dev_X,dev_Y))
	print 'Number of support vectors:', sum(clf.n_support_)
	support_vectors = clf.support_vectors_
	# 4
	test_X, test_Y = test_data
	test_Y = clf.predict(test_X)
	return test_Y

def test_predict(filename_r, filename_w, test):
	f = open(filename_r)
	lines = f.readlines()
	count = 0
	positive_count = 0
	output = ''
	for i in test_Y:
		if i > 0:
			l = lines[count][:-2].split(', ')
			l.append('>50')
			positive_count += 1
		else:
			l = lines[count][:-2].split(', ')
			l.append('<=50')
		l_str = ', '.join(l)
		l_str += '\n'
		output += l_str
		count += 1
	positive_rate = positive_count/count
	print 'test_positive_rate:', '{:.2%}'.format(positive_rate)
	ft = open(filename_w,'w')
	ft.write(output)
	ft.close()


if __name__ == '__main__':
	# input C
	C = float(sys.argv[1])
	train_sample_size = 5000
	feature2index = create_feature_map('income.train.txt.5k')
	dimension = len(feature2index)
	train_data = map_data('income.train.txt.5k', feature2index)
	dev_data = map_data('income.dev.txt', feature2index)

	model, support_vectors = train(train_data, dev_data, C)
	w, b = model

	# 1.2
	train_X, train_Y = train_data
	feature_lable = {}
	for (x,y) in zip(train_X, train_Y):
		feature_lable[tuple(x)] = y

	sv_violate_count = 0
	for sv in support_vectors:
		y = feature_lable[tuple(sv)]
		if y * (np.dot(w, sv) + b) < 1:
			sv_violate_count += 1
	print 'Number of support vectors have margin violations:', sv_violate_count

	# 1.3 & 1.4
	total_slacks = 0
	slack_list = []
	for i, (x,y) in enumerate(zip(train_X, train_Y)):
		temp = max(0, 1 - y*(np.dot(w, x) + b))
		slack_list.append((round(y * temp,2), i))
		total_slacks += temp
	print '-----1.3-----'
	print 'Total amount of margin violations:', round(total_slacks,2)

	total_objective = 1/2 * np.dot(w,w) + C * total_slacks
	print 'Total objective:', round(total_objective,2)

	slack_list.sort()
	positive_slack = slack_list[-5:]
	negative_slack = slack_list[:5]
	print '-----1.4-----'
	print 'Top five most violated training examples in positive class:', positive_slack
	print 'Top five most violated training examples in negative class:', negative_slack

	# 2
	# input Epoch
	print '----- 2 -----'
	Epoch = int(sys.argv[2])
	pegasos_w = pegasos(train_data, dev_data, Epoch, C)
	pegasos_support_vector = 0
	for x,y in zip(train_data[0],train_data[1]):
		if y * np.dot(pegasos_w, x) <= 1:
			pegasos_support_vector += 1
	print 'pegasos support vector', pegasos_support_vector
	

	# 3
	print '----- 3 -----'
	# C = int(sys.argv[3])
	test_data = map_data('income.test.txt', feature2index)
	test_Y = quad_train(train_data, dev_data, test_data, C)
	# 4
	print '----- 4 -----'
	test_predict('income.test.txt', 'income.test.predicted', test_Y)

