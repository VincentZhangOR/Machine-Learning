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
	
def aver_perceptron(max_epoch, filename,templates,dictionary):
	count = 1
	best_err_rate = float('inf')
	w = defaultdict(int)
	# best_w = defaultdict(int)
	aver_w = defaultdict(float)
	last = []
	w_prime = defaultdict(float)
	train_err = []
	dev_err = []
	for i in xrange(max_epoch):
		update = 0
		for words, tags in readfile(filename):
			mytags = tri_decode(words, dictionary, w)
			if mytags != tags:
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
			aver_w[key] += w[key] - w_prime[key]/count
		# print aver_w
		cur = [x for x in aver_w.keys() if aver_w[x] != 0.0]
		print cur == last
		last = deepcopy(cur)
				
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

	dictionary, model = mle(trainfile)
	
	max_epoch = 5
	t0 = time()
	#w_perceptron, unaver_train_err, unaver_dev_err= perceptron(max_epoch, trainfile)
	t1 = time()
	w_aver_perceptron, aver_train_err,aver_dev_err = aver_perceptron(max_epoch, trainfile,'tri',dictionary)
	t2 = time()
	
	print 'unaverage percepron ', t1 - t0, 'average percepron ', t2 - t1
