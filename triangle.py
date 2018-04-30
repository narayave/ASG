'''
Demo for ASG

Author:
	Wei-Yang Qu  quwy@lamda.nju.edu.cn
Date:
	2018.04.15
'''

import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC, OneClassSVM
from asg import ASG
from class_filter import ClassFilter
from components import *
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.utils import shuffle

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pprint import pprint


def split_dataset(train, test, orig):
	train_dataX, train_dataY = [], []
	test_dataX, test_dataY = [], []
	orig_x = []

	for i in range(0, len(orig)):
		orig_x.append(orig[i, :6])

	for i in range(0, len(train)):
		# print data[i, :10]
		train_dataX.append(train[i, :6])
		train_dataY.append(train[i, 6:][0])

	for i in range(0, len(test)):
		test_dataX.append(test[i, :6])
		test_dataY.append(test[i, 6:][0])

	scale = MinMaxScaler(feature_range=(0, 1))
	orig_norm = scale.fit_transform(orig_x)
	trainX_norm = scale.transform(train_dataX)
	testX_norm = scale.transform(test_dataX)

	joblib.dump(scale, 'scale.pkl')

	# trainX = np.unique(np.array(trainX_norm), axis=0)
	trainX = np.array(trainX_norm)
	trainY = np.array(train_dataY)
	testX = np.array(testX_norm)
	# testX = np.unique(np.array(testX_norm), axis=0)
	testY = np.array(test_dataY)

	return trainX_norm, train_dataY, testX_norm, test_dataY


def create_dataset(train_file, test_file, orig):
	dataset = []

	dataframe = read_csv(orig, engine='python')
	# print len(dataframe)
	orig = dataframe.values

	# TODO: Changing input here
	dataframe = read_csv(train_file, engine='python')
	# print len(dataframe)
	train = dataframe.values
	train = shuffle(train)

	dataframe = read_csv(test_file, engine='python')
	# print len(dataframe)
	test = dataframe.values

	print len(train), len(test)

	train_X, train_Y, test_X, test_Y = split_dataset(train, test, orig)

	print 'Training data details: ', str(len(train_X)), str(len(train_Y))
	print 'Test data details:', str(len(test_X)), str(len(test_Y))
	return train_X, train_Y, test_X, test_Y



'''
	PCA to reduce dimensionality and be able to graph
'''
def reduce_graph(train_X, test_X):

	print 'train X', train_X.shape
	print 'test X', test_X.shape

	pca = PCA(n_components=2)
	pca.fit(train_X)

	train = pca.transform(train_X)
	test = pca.transform(test_X)
	joblib.dump(pca, 'pca.pkl')

	print pca
	print pca.explained_variance_ratio_
	print pca.singular_values_
	print pca.mean_

	return train, test


def graph(train, train_y, test, test_y):


	# space_sampling_points = 100
	# # Generate a regular grid to sample the 3D space for various operations later
	# xx, yy = np.meshgrid(np.linspace(-3, 3, space_sampling_points),
	# 					np.linspace(-3, 3, space_sampling_points))


	# # plot the line, the points, and the nearest vectors to the plane
	# Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
	# Z = Z.reshape(xx.shape)

	fig = plt.figure()
	# ax = fig.add_subplot(111, projection='3d')

	# print 'About to graph'

	# plt.title("Novelty Detection")
	# plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 1, 7), cmap=plt.cm.PuBu)
	# a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
	# plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')

	s = 20

	plt.scatter(train[:, 0], train[:, 1], c=train_y, cmap='autumn') #, s=s)
	plt.scatter(test[:, 0], test[:, 1], c=test_y, cmap='spring') #, s=s)


	# plt.savefig("pca_svm.svg", dpi=3600, format='svg')
	plt.show()

def graph_3d(train, train_y, test, test_y):

	fig = plt.figure()
	ax = fig.gca(projection='3d')

	s = 20
	b1 = ax.scatter(train[:, 0], train[:, 1], train[:, 2], c=train_y, cmap='autumn')
	b2 = ax.scatter(test[:, 0], test[:, 1], test[:, 2], c=test_y, cmap='spring')

	plt.show()


'''
Procedure:
		Input the origin data, and filter data by category.
		Generate positive and negative data of each class.
		Use ASG method to get the margin of each class.
		Input test data, use the model given by Asg to decide
			which class the test data should be.
'''
if __name__ == '__main__':

	train = 'data/mixed_smalllarge_002.csv'
	test = 'data/all_003.csv'

	orig = 'data/original_30.csv'

	train_X, train_Y, test_X, test_Y = create_dataset(train, test, orig)

	seen_class = [0]

	# seen class can be set here! default seen class is all class
	#  in train data
	cf = ClassFilter(train_X, train_Y, SeenClass = seen_class)
	data_by_label = cf.Filter()
	print "Total label:", cf.getDistinctLabel()
	print "Seen class is:", seen_class

	'''
	# classifier model in ASG can be set here.
	# Note that classifier needs to support the parameter
	# 	'sample_weight' in the 'fit' function.
	# Otherwise, you need to modify the function 'train_Dminus'
	# 	and 'train_Dplus' in the file gen_data.py
	'''
	classifier_model = SVC(kernel='rbf', gamma=5, probability = True)
	# classifier_model = SVC(kernel='rbf', probability = True)

	# ASG method: initial
	asg = ASG(classifier=classifier_model, classfilter = cf)

	# run_ASG:
	# generate_size: the size of the sample you want to generate
	# sample_size: sample size in origin data when generating data
	asg.run_ASG(generate_size = 50, sample_size = 50)

	# predict for the test data with unseen class. If the test data
	#  belongs to unseen class, then output -1
	print("[ASG] performance on test data")
	train_preds = asg.predict(train_X)
	test_preds = asg.predict(test_X)
	# set unseen label to -1
	test_label = dealTesty(test_Y, seen_class)
	# print test_label[p]
	get_macroF1(test_preds, test_label)

	pprint(train_preds[:len(train_preds)])
	n_error_train = train_preds[train_preds == -1].size
	n_error_test = test_preds[test_preds == -1].size
	print n_error_train, n_error_test

	print 'train - ', 100*float(n_error_train)/train_X.shape[0]
	print 'test - ', 100*float(n_error_test)/test_X.shape[0]

	train_X, test_X = reduce_graph(train_X, test_X)
	graph(train_X, train_preds, test_X, test_preds)
	# graph_3d(train_X, train_preds, test_X, test_preds)