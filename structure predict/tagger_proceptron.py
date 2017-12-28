#!/usr/bin/env python

from __future__ import division
from collections import defaultdict
import sys
import matplotlib.pyplot as plt
from tagger1 import *
from time import *
from copy import deepcopy

startsym, stopsym = "<s>", "</s>"

trainfile = 'train.txt.lower.unk'
devfile = 'dev.txt.lower.unk'
testfile = 'test.txt.lower.unk.unlabeled'


def feature(words, tags):
	phi = defaultdict(int)
	tags = [startsym] + tags + [stopsym]
	words = [startsym] + words + [stopsym]
	#print len(words),len(tags)
	for i in xrange(len(words)-1):
		#print i
		phi[tags[i],tags[i+1]] += 1
	for i in xrange(len(words)):
		phi[tags[i], words[i]] += 1
	return phi
	
def tri_feature(words, tags):
	phi = defaultdict(int)
	tags = [startsym] + [startsym] + tags + [stopsym] + [stopsym]
	words = [startsym] + [startsym] + words + [stopsym] + [stopsym]
	#print len(words),len(tags)
	for i in xrange(len(tags)-2):
		#print i
		#phi[tags[i+2]] += 1
		#phi[tags[i+1],tags[i+2]] += 1
		phi[tags[i],tags[i+1],tags[i+2]] +=1
	for i in xrange(len(tags)-1):
		phi[tags[i], tags[i+1]] += 1
	for i in xrange(len(words)):
		phi[tags[i], words[i]] += 1
	for i in xrange(len(words)):
		phi[tags[i]] += 1
	return phi

def perceptron(max_epoch, filename):
	best_err_rate = float('inf')
	w = defaultdict(int)
	#features = set()
	best_w = w
	train_err = []
	dev_err = []
	for i in xrange(max_epoch):
		update = 0
		for words, tags in readfile(filename):
			mytags = decode(words, dictionary, w)
			if mytags != tags:
				phi_xy = feature(words, tags)
				phi_xz = feature(words, mytags)
				for key in set(phi_xy.keys() + phi_xz.keys()):
					w[key] += phi_xy[key]
					w[key] -= phi_xz[key]
					#features.add(key)
				update += 1
		train_err_rate = test(trainfile, dictionary, w)
		train_err.append(train_err_rate)
		dev_err_rate = test(devfile, dictionary, w)
		dev_err.append(dev_err_rate)
		if best_err_rate > dev_err_rate:
			best_err_rate = dev_err_rate
			best_w = w
		
		print 'epoch ', str(i+1) + ',', 'update ', str(update) + ',', 'features', str(len(w)) + ',', 'train_err {0:.2%}'.format(train_err_rate), 'dev_err {0:.2%}'.format(dev_err_rate)
	print 'best_err {0:.2%}'.format(best_err_rate)
	#print features
	
	return best_w, train_err, dev_err
	
def aver_perceptron(max_epoch, filename,templates):
	count = 1
	best_err_rate = float('inf')
	w = defaultdict(int)
	# best_w = defaultdict(int)
	aver_w = defaultdict(float)
	last = aver_w
	w_prime = defaultdict(float)
	train_err = []
	dev_err = []
	for i in xrange(max_epoch):
		update = 0
		for words, tags in readfile(filename):
			if templates == 'bi':
				mytags = decode(words, dictionary, w)
			else:
				mytags = tri_decode(words, dictionary, w)
			if mytags != tags:
				if templates == 'bi':
					phi_xy = feature(words, tags)
					phi_xz = feature(words, mytags)
				else:
					phi_xy = tri_feature(words, tags)
					phi_xz = tri_feature(words, mytags)
				for key in set(phi_xy.keys() + phi_xz.keys()):
					w[key] += phi_xy[key]
					w[key] -= phi_xz[key]
					w_prime[key] += count*phi_xy[key]
					w_prime[key] -= count*phi_xz[key]
				update += 1
			count += 1
		#print w
		for key in set(w.keys() + w_prime.keys()):
			aver_w[key] = w[key] - w_prime[key]/count
		# print aver_w
		print aver_w == last
		last = deepcopy(aver_w)
				
		train_err_rate = test(trainfile, dictionary, aver_w)
		train_err.append(train_err_rate)
		dev_err_rate = test(devfile, dictionary, aver_w)
		dev_err.append(dev_err_rate)
		if best_err_rate > dev_err_rate:
			best_err_rate = dev_err_rate
			best_w = aver_w
		
		print 'epoch ', str(i+1) + ',', 'update ', str(update) + ',', 'features', str(len(w)) + ',', 'train_err {0:.2%}'.format(train_err_rate), 'dev_err {0:.2%}'.format(dev_err_rate)
	print 'best_err {0:.2%}'.format(best_err_rate)
	
	return (best_w, train_err, dev_err)
        
if __name__ == "__main__":
	# trainfile, devfile = sys.argv[1:3]

	dictionary, model = mle(trainfile)
	
	max_epoch = 5
	t0 = time()
	#w_perceptron, unaver_train_err, unaver_dev_err= perceptron(max_epoch, trainfile)
	t1 = time()
	w_aver_perceptron, aver_train_err,aver_dev_err = aver_perceptron(max_epoch, trainfile,'tri')
	t2 = time()
	
	print 'unaverage percepron ', t1 - t0, 'average percepron ', t2 - t1

	x = [x + 1 for x in xrange(max_epoch)]
	#print x
	#plt.plot(x, unaver_train_err, 'b', x, unaver_dev_err, 'b--', x, aver_train_err, 'r', x, aver_dev_err, 'r--')
	#plt.show()
	#print "train_err {0:.2%}".format(test(trainfile, dictionary, model))
	#print "dev_err {0:.2%}".format(test(devfile, dictionary, model))
