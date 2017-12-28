#!/usr/bin/env python

from collections import defaultdict

def dict_add(d1, d2):
	d = defaultdict(float)
	for key in set(d1.keys()+d2.keys()):
		d[key] = d1[key] + d2[key]
	return d

def dict_sub(d1, d2):
	d = defaultdict(float)
	for key in set(d1.keys()+d2.keys()):
		d[key] = d1[key] - d2[key]
	return d

def dict_mul(d1, c):
	d = defaultdict(float)
	for key in d1.keys():
		d[key] = d1[key] * c
	return d