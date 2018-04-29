'''
Demo for ASG

Author:
	Wei-Yang Qu  quwy@lamda.nju.edu.cn
Date:
	2018.04.15
'''

import numpy as np
from sklearn.svm import SVC
from asg import ASG
from class_filter import ClassFilter
from components import *
from pandas import read_csv

from sklearn.preprocessing import MinMaxScaler

def split_dataset(train, test):
	train_dataX, train_dataY = [], []
	test_dataX, test_dataY = [], []


	for i in range(0, len(train)):
		# print data[i, :10]
		train_dataX.append(train[i, :6])
		train_dataY.append(train[i, 6:][0])

	for i in range(0, len(test)):
		test_dataX.append(test[i, :6])
		test_dataY.append(test[i, 6:][0])

	scale = MinMaxScaler(feature_range=(0, 1))
	trainX_norm = scale.fit_transform(train_dataX)
	testX_norm = scale.transform(test_dataX)

	# joblib.dump(scale, 'scale.pkl')

	# trainX = np.unique(np.array(trainX_norm), axis=0)
	trainX = np.array(trainX_norm)
	trainY = np.array(train_dataY)
	testX = np.array(testX_norm)
	# testX = np.unique(np.array(testX_norm), axis=0)
	testY = np.array(test_dataY)

	return trainX_norm, train_dataY, testX_norm, test_dataY


def create_dataset(train_file, test_file):
	dataset = []

	# TODO: Changing input here
	dataframe = read_csv(train_file, engine='python')
	# print len(dataframe)
	train = dataframe.values

	dataframe = read_csv(test_file, engine='python')
	# print len(dataframe)
	test = dataframe.values


	# train_size = int(len(dataset) * 0.8)
	# test_size = len(dataset) - train_size
	# train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
	print len(train), len(test)

	train_X, train_Y, test_X, test_Y = split_dataset(train, test)

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
	# joblib.dump(pca, 'pca.pkl')

	print pca
	print pca.explained_variance_ratio_
	print pca.singular_values_
	print pca.mean_

	return train, test



'''
Procedure:
		Input the origin data, and filter data by category.
		Generate positive and negative data of each class.
		Use ASG method to get the margin of each class.
		Input test data, use the model given by Asg to decide which class the test data should be.
'''
if __name__ == '__main__':

	train = 'good.csv'
	test = 'good.csv'

	train_X, train_Y, test_X, test_Y = create_dataset(train, test)

	seen_class = [0]

	# seen class can be set here! default seen class is all class in train data
	cf = ClassFilter(train_X, train_Y, SeenClass = seen_class)
	data_by_label = cf.Filter()
	print "Total label:", cf.getDistinctLabel()
	print "Seen class is:", seen_class

	'''
	# classifier model in ASG can be set here.
	# Note that classifier needs to support the parameter 'sample_weight' in the 'fit' function.
	# Otherwise, you need to modify the function 'train_Dminus' and 'train_Dplus' in the file gen_data.py
	'''
	classifier_model = SVC(kernel='rbf', probability = True)

	# ASG method: initial
	asg = ASG(classifier=classifier_model, classfilter = cf)

	# run_ASG:
	# generate_size: the size of the sample you want to generate
	# sample_size: sample size in origin data when generating data
	asg.run_ASG(generate_size = 50, sample_size = 100)

	# predict for the test data with unseen class. If the test data belongs to unseen class, then output -1
	print("[ASG] performance on test data")
	train_preds = asg.predict(train_X)
	test_preds = asg.predict(test_X)

	# set unseen label to -1
	test_label = dealTesty(test_Y, seen_class)
	print test_label
	get_macroF1(test_preds, test_label)


	train, test = reduce_graph(train_X, test_X)

	graph(train, train_y, )