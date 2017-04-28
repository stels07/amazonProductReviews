from sen2vec import skipthoughts
import matplotlib.pyplot as plt
import numpy as np
import json
from nltk import tokenize
from sklearn import manifold
import os
from flask import Flask, request, jsonify
from flask.ext.restful import Resource, Api, reqparse
from gensim.models.word2vec import Word2Vec as w
from gensim import utils, matutils
import cPickle
import argparse
import base64
import sys



def read_sentences(filename):
	z = json.loads(open(filename, "r").read())
	data = []
	ID_set = []
	for i in range(0, len(z)):
		ID = z[i].get("asin")
		print ID
		# print nn
		data.append([ID, nn])
		# print data[0]
		ID_set.append(ID)
	ID_set = list(set(ID_set))

	dic = {}
	for i in ID_set:
		dic[i] = []

	for i in data:
		for j in i[1]:
			dic[i[0]].append(j)


	return dic, list(ID_set)

def sen2vec(model, sentences):
	# just pretrained
	return skipthoughts.encode(model, sentences)

def smy2vec(sentences):
	words = sentences.split(' ')
	return np.mean(word2vec(sentences), axis=0)


def reduce_dimension(X):
	# to do: 4800 dimension to 2 dimension
	from time import time
	t0 = time()
	tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
	trans_data = tsne.fit_transform(X).T
	t1 = time()
	print("t-SNE: %.2g sec" % (t1 - t0))

	return np.transpose(trans_data)

def demo_front_end(sentences, positions):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.axis([-20, 20, -20, 20])
	l = len(sentences)
	for i in range(l):
		s = sentences[i]
		p = positions[i]
		ax.text(p[0], p[1], s)

	plt.show()

def front_end(sentences, positions):
	pass

def load_json(ID, X, p):
	results = []
	for index in range(0, len(X)):
		result = {}
		result["text"] = X[index]
		result["x"] = p[index][0]
		result["y"] = p[index][1]
		results.append(result)

	print results[0]

	with open('result/' + ID + '.json', 'w') as fp:
		json.dump(results, fp, ensure_ascii=False, separators=(',', ':'))

def read_json(ID):
	filename = 'result/' + ID + '.json'
	z = json.loads(open(filename, "r").read())
	return [t.get("text") for t in z]


if __name__ == '__main__':

	script_dir = os.path.dirname(__file__) #<-- absolute dir the script is i
	rel_path = 'GoogleNews-vectors-negative300.bin'
	abs_file_path = os.path.join(script_dir, rel_path)
	model = w.load_word2vec_format(abs_file_path, binary=True)

	ID = 'B00004R8VM'
	x = read_json(ID)

	vecs = []
	for i in x:
		words = i.split(' ')
		vec = []
		for j in words:
			try:
				vec.append(model[j])
			except:
				pass
		tmp = np.mean(vec, axis=0)
		vecs.append(tmp)

	load_json(ID, x, reduce_dimension(vecs))

	# load_json(ID, )
	# print dic
	# print ID_list

	# model = skipthoughts.load_model()

	# Id = ID_list[50]
	# for ID in ID_list:
	# 	p = reduce_dimension(sen2vec(model, dic[ID]))  
	# 	load_json(ID, dic[ID], p)
	# 	if ID == Id:
	# 		demo_front_end(dic[ID], p)
	


	






