from __future__ import division
from collections import defaultdict
import numpy as np
import math
from random import shuffle
import time

# mode = 'perceptron'
# mode = 'naive_average'
mode = 'smart_average'
# mode = 'non_aggressibe_mira'
mode = 'non_aggressibe_average_mira'
# mode = 'aggressibe_mira'

isBinned = 'binarize_non_binned'
# isBinned = 'binarize'
isRescale = 'Yes'
# isRescale = 'No'

temp_l = [None for x in xrange(9)]
temp_l[0] = ['17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '88', '90']
temp_l[1] = ['Federal-gov', 'Local-gov', 'Private', 'Self-emp-inc', 'Self-emp-not-inc', 'State-gov', 'Without-pay']
temp_l[2] = ['10th', '11th', '12th', '1st-4th', '5th-6th', '7th-8th', '9th', 'Assoc-acdm', 'Assoc-voc', 'Bachelors', 'Doctorate', 'HS-grad', 'Masters', 'Preschool', 'Prof-school', 'Some-college']
temp_l[3] = ['Divorced', 'Married-AF-spouse', 'Married-civ-spouse', 'Married-spouse-absent', 'Never-married', 'Separated', 'Widowed']
temp_l[4] = ['Adm-clerical', 'Armed-Forces', 'Craft-repair', 'Exec-managerial', 'Farming-fishing', 'Handlers-cleaners', 'Machine-op-inspct', 'Other-service', 'Priv-house-serv', 'Prof-specialty', 'Protective-serv', 'Sales', 'Tech-support', 'Transport-moving']
temp_l[5] = ['Amer-Indian-Eskimo', 'Asian-Pac-Islander', 'Black', 'Other', 'White']
temp_l[6] = ['Female', 'Male']
temp_l[7] = ['1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '3', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '4', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '5', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '6', '60', '61', '62', '63', '64', '65', '66', '67', '68', '7', '70', '72', '73', '74', '75', '76', '77', '78', '8', '80', '81', '82', '84', '85', '86', '87', '88', '89', '9', '90', '91', '92', '94', '95', '96', '97', '98', '99']
temp_l[8] = ['Cambodia', 'Canada', 'China', 'Columbia', 'Cuba', 'Dominican-Republic', 'Ecuador', 'El-Salvador', 'England', 'France', 'Germany', 'Greece', 'Guatemala', 'Haiti', 'Holand-Netherlands', 'Honduras', 'Hong', 'Hungary', 'India', 'Iran', 'Ireland', 'Italy', 'Jamaica', 'Japan', 'Laos', 'Mexico', 'Nicaragua', 'Outlying-US(Guam-USVI-etc)', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Puerto-Rico', 'Scotland', 'South', 'Taiwan', 'Thailand', 'Trinadad&Tobago', 'United-States', 'Vietnam', 'Yugoslavia']
# temp_l[9] = ['<=50K', '>50K']

categary = defaultdict(dict)
for i in xrange(9):
	for j,x in enumerate(temp_l[i]):
		categary[i][x] = j
categary[9] = {'<=50K':-1, '>50K':1}

reverse = []
for i in xrange(9):
	reverse += temp_l[i]

def binarize_non_binned(file_name):
	f = open(file_name)
	lines = f.readlines()
	matrix = []
	for line in lines:
		vector = []
		l = line[:-1].split(', ')
		for i in xrange(10):
			cur = l[i]
			if i == 9:
				vector.append(categary[i][cur])
			else:
				temp = [-1 for x in xrange(len(categary[i]))]
				temp[categary[i][cur]] = 1
				vector += temp		
		matrix.append(vector)
	return matrix

def binarize(file_name):
	f = open(file_name)
	lines = f.readlines()
	matrix = []
	for line in lines:
		vector = []
		l = line[:-1].split(', ')
		for i in xrange(10):
			cur = l[i]
			temp = []
			if i == 0:
				for j in xrange(1, 9):
					if 10 * j < int(cur) <= 10 * j + 10:
						temp.append(1)
					else:
						temp.append(-1)
			elif i == 7:
				for j in xrange(10):
					if 10 * j < int(cur) <= 10 * j + 10:
						temp.append(1)
					else:
						temp.append(-1)
			elif i == 9:
				vector.append(categary[i][cur])
			else:
				temp = [-1 for x in xrange(len(categary[i]))]
				temp[categary[i][cur]] = 1
			vector += temp		
		matrix.append(vector)
	return matrix

def rescale(matrix):
	col = [0 for x in xrange(len(matrix[0])-1)]
	for sample in matrix:
		for i in xrange(len(sample)-1):
			col[i] += sample[i]
	norm = [[0,0] for x in xrange(len(matrix[0])-1)]
	for i, x in enumerate(col):
		E_x = x / len(matrix)
		D_x = math.sqrt(1 - E_x ** 2)
		if D_x > 0.0:
			norm[i][0] = (1 - E_x) / D_x
			norm[i][1] = (-1 - E_x) / D_x
	for sample in matrix:
		for i in xrange(len(sample)-1):
			if D_x > 0.0:
				sample[i] = norm[i][0] if sample[i] == 1 else norm[i][1]
	print matrix
	return matrix

def dev(epoch, w, b):
	np_matrix_dev = np.array(matrix_dev)
	total_count = 0
	err_count = 0	
	for sample in np_matrix_dev:
		total_count += 1
		x = sample[:-1]
		y = sample[-1]
		if y * (np.dot(w,x) + b) <= 0:
			err_count += 1
	error_rate = float(err_count)/total_count
	# print 'epoch:', epoch, 'Error Rate:', error_rate
	return error_rate

def perceptron(max_epoch, w, b):
	count = 0
	global min_error_rate
	for i in xrange(max_epoch):
		# shuffle(np_matrix_train)
		for sample in np_matrix_train:
			count += 1
			x = sample[:-1]
			y = sample[-1]
			if y * (np.dot(w,x) + b) <= 0:
				w = w + y * x
				b = b + y
			if count % 1000 == 0:
				epoch = count/len(matrix_train)
				temp_error_rate = dev(epoch, w, b)
				if temp_error_rate < min_error_rate[1]:
					min_error_rate = (epoch, temp_error_rate)
	return (w, b)

def naive_average(max_epoch, w, b):
	count = 0
	global min_error_rate
	w_prime = np.array([0.0 for x in xrange(len(matrix_train[0])-1)])
	b_prime = 0.0
	for i in xrange(max_epoch):
		# shuffle(np_matrix_train)
		for sample in np_matrix_train:
			count += 1
			x = sample[:-1]
			y = sample[-1]
			if y * (np.dot(w,x) + b) <= 0:
				w = w + y * x
				b = b + y
			w_prime += w
			b_prime += b
			if count % 1000 == 0:
				epoch = count/len(matrix_train)
				temp_error_rate = dev(epoch, w_prime/count, b_prime/count)
				if temp_error_rate < min_error_rate[1]:
					min_error_rate = (epoch, temp_error_rate)
	return (w_prime/count, b_prime/count)

def smart_average(max_epoch, w, b):
	count = 0
	global min_error_rate
	w_prime = np.array([0.0 for x in xrange(len(matrix_train[0])-1)])
	b_prime = 0.0
	for i in xrange(max_epoch):
		# shuffle(np_matrix_train)
		for sample in np_matrix_train:
			count += 1
			x = sample[:-1]
			y = sample[-1]
			if y * (np.dot(w,x) + b) <= 0:
				w = w + y * x
				b = b + y
				w_prime += count * y * x
				b_prime += count * y
			if count % 1000 == 0:
				epoch = count/len(matrix_train)
				temp_error_rate = dev(epoch, w - w_prime/count, b - b_prime/count)
				if temp_error_rate < min_error_rate[1]:
					min_error_rate = (epoch, temp_error_rate)
	return (w - w_prime/count, b - b_prime/count)

def non_aggressibe_mira(max_epoch, w, b):
	count = 0
	global min_error_rate
	for i in xrange(max_epoch):
		# shuffle(np_matrix_train)
		for sample in np_matrix_train:
			count += 1
			x = sample[:-1]
			y = sample[-1]
			if y * (np.dot(w,x) + b) <= 0:
				w = w + (y - np.dot(w,x))/np.dot(x,x) * x
				b = b + (y - np.dot(w,x))/np.dot(x,x)
			if count % 1000 == 0:
				epoch = count/len(matrix_train)
				temp_error_rate = dev(epoch, w, b)
				if temp_error_rate < min_error_rate[1]:
					min_error_rate = (epoch, temp_error_rate)
	return (w, b)

def non_aggressibe_average_mira(max_epoch, w, b):
	count = 0
	global min_error_rate
	w_prime = np.array([0.0 for x in xrange(len(matrix_train[0])-1)])
	b_prime = 0.0
	for i in xrange(max_epoch):
		# shuffle(np_matrix_train)
		for sample in np_matrix_train:
			count += 1
			x = sample[:-1]
			y = sample[-1]
			if y * (np.dot(w,x) + b) <= 0:
				w = w + (y - np.dot(w,x))/np.dot(x,x) * x
				b = b + (y - np.dot(w,x))/np.dot(x,x)
				# w_prime += count * (y - np.dot(w,x))/np.dot(x,x) * x
				# b_prime += count * (y - np.dot(w,x))/np.dot(x,x)
			w_prime += w
			b_prime += b
			if count % 1000 == 0:
				epoch = count/len(matrix_train)
				temp_error_rate = dev(epoch, w_prime/count, b_prime/count)
				if temp_error_rate < min_error_rate[1]:
					min_error_rate = (epoch, temp_error_rate)
	return (w_prime/count, b_prime/count)

def aggressibe_mira(max_epoch, w, b, p):
	count = 0
	global min_error_rate
	for i in xrange(max_epoch):
		# shuffle(np_matrix_train)
		for sample in np_matrix_train:
			count += 1
			x = sample[:-1]
			y = sample[-1]
			if y * (np.dot(w,x) + b) <= p:
				w = w + (y - np.dot(w,x))/np.dot(x,x) * x
				b = b + (y - np.dot(w,x))/np.dot(x,x)
			if count % 1000 == 0:
				epoch = count/len(matrix_train)
				temp_error_rate = dev(epoch, w, b)
				if temp_error_rate < min_error_rate[1]:
					min_error_rate = (epoch, temp_error_rate)
	return (w, b)


############### main body ###################

if isBinned == 'binarize_non_binned':
	matrix_train = binarize_non_binned("income.train.txt")  # binarize data in training set
	matrix_dev = binarize_non_binned("income.dev.txt")  # binarize data in dev set
else:
	matrix_train = binarize("income.train.txt")  # binarize data in training set
	matrix_dev = binarize("income.dev.txt")  # binarize data in dev set

if isRescale == 'Yes':
	matrix_train = rescale(matrix_train)  # rescale vector in matrix_train
	matrix_dev = rescale(matrix_dev)  # rescale vector in matrix_dev

np_matrix_train = np.array(matrix_train)
w = np.array([0.0 for x in xrange(len(matrix_train[0])-1)])
b = 0.0
max_epoch = 5
min_error_rate = (0, float('inf'))

# training
if mode == 'perceptron':
	(w, b) = perceptron(max_epoch, w, b)   # 16.777%
if mode == 'naive_average':
	start = time.time()
	(w, b) = naive_average(max_epoch, w, b)   # 15.716%
	end = time.time()
	print 'running time:', end - start
if mode == 'smart_average':
	start = time.time()
	(w, b) = smart_average(max_epoch, w, b)   # 15.716%
	end = time.time()
	print 'running time:', end - start   # running time: 1.24909090996 (without printing)
if mode == 'non_aggressibe_mira':
	(w, b) = non_aggressibe_mira(max_epoch, w, b)   # epoch: 1.21569349788 MinErrorRate 0.165782493369
if mode == 'non_aggressibe_average_mira':
	(w, b) = non_aggressibe_average_mira(max_epoch, w, b)  # epoch: 3.05765334316 MinErrorRate 0.157824933687
if mode == 'aggressibe_mira':
	# (p=0.4) 2.21035181433 MinErrorRate 0.159151193634
	p_list = [0.1,0.5,0.9]
	for p in p_list:
		(w, b) = aggressibe_mira(max_epoch, w, b, p)   
		# (p=0.1) epoch: 0.221035181433 MinErrorRate 0.173076923077
		# (p=0.5) epoch: 2.21035181433 MinErrorRate 0.163793103448
		# (p=0.9) epoch: 3.02081414625 MinErrorRate 0.161140583554
		print 'epoch:', min_error_rate[0], 'MinErrorRate', min_error_rate[1]
print 'final result:'
print 'epoch:', min_error_rate[0], 'MinErrorRate', min_error_rate[1]
if isBinned == 'binarize_non_binned':
	sort_w_index = sorted(zip(w, xrange(len(matrix_train[0])-1)))
	positive_five = sort_w_index[::-1][:5]
	nagetive_five = sort_w_index[:5]
	for i in xrange(5):
		w_pos, index_pos = positive_five[i]
		w_nag, index_nag = nagetive_five[i]
		positive_five[i] = (w_pos, reverse[index_pos])
		nagetive_five[i] = (w_nag, reverse[index_nag])
	print positive_five
	print nagetive_five
