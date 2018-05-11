'''
	Graphing tool to view generated data points

	Author:
		Vedanth Narayanan	narayave@oregonstate.edu
	Date:
		2018.04.28
'''

import numpy as np

import matplotlib.pyplot as plt
from pandas import read_csv
import pandas as pd
import csv

from pprint import pprint
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.externals import joblib


SCALER = MinMaxScaler(feature_range=(0, 1))
PCA_REDUCE = None #PCA(n_components=2)

def concatenate(in_files, out_file):

	with open(out_file, 'w') as outfile:
		for fname in in_files:
			with open(fname) as infile:
				for line in infile:
					outfile.write(line)


def load_modules():

	global SCALER
	global PCA_REDUCE

	SCALER = joblib.load("/home/ved/ASG/scale.pkl")
	PCA_REDUCE = joblib.load("/home/ved/ASG/pca.pkl")

	print 'Modules loaded. Over.'


def to_dataframe(filename):

	with open(filename) as f:
		lines = f.readlines()
	# pprint(lines)

	lines = [x.split(' \n')[0] for x in lines]
	# pprint(lines)
	lines = [map(float, x.split()) for x in lines]
	# pprint(lines)

	return lines


def transform(data):


	scaled = SCALER.transform(data)
	scaled = np.array(scaled)

	pca_ed = PCA_REDUCE.transform(scaled)

	# Ready for graping
	all_data = pca_ed
	return all_data


def graph(minus, plus):
	fig = plt.figure()

	s = 20
	plt.scatter(minus[:, 0], minus[:, 1], s=s, c='b')
	plt.scatter(plus[:, 0], plus[:, 1], s=s, c='y')

	# plt.savefig("graph_tool.svg", dpi=3600, format='svg')
	plt.show()


def graph_3d(minus, plus):

	fig = plt.figure()
	ax = fig.gca(projection='3d')

	s = 20
	b1 = ax.scatter(minus[:, 0], minus[:, 1], minus[:, 2], s=s, c='b')
	b2 = ax.scatter(plus[:, 0], plus[:, 1], plus[:, 2], s=s, c='y')

	plt.show()



if __name__ == '__main__':

	in_files = ['gendata/D_minus0_1', 'gendata/D_plus0_1']
	# out_file = 'gendata/all_data'

	load_modules()
	minus = transform(to_dataframe(in_files[0]))
	plus = transform(to_dataframe(in_files[1]))
	graph(minus, plus)
	# graph_3d(minus, plus)