from __future__ import absolute_import
#from __future__ import print_function
import numpy as np

import csv

import random
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Input, Lambda
from keras.optimizers import RMSprop
from keras import backend as K

from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.constraints import maxnorm

from time import time
from PIL import Image

from imageDataExtract import *


breakLimit = 6
rowLimit = 500
modelPath = '../models/model0.h5'


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))



def create_base_network(input_dim):

	model = Sequential()

	model.add(Convolution2D(32, 3, 3, input_shape=(3, 64, 64), border_mode='same', activation='relu', W_constraint=maxnorm(3)))
	model.add(Activation('relu'))
	#model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	
	model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu', W_constraint=maxnorm(3)))
	model.add(Activation('relu'))
	#model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))

	model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu', W_constraint=maxnorm(3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	
	model.add(Flatten())
	model.add(Dense(512, activation='relu', input_dim=(3,64,64), W_constraint=maxnorm(3)))
	model.add(Dense(64, activation='relu', W_constraint=maxnorm(3)))

	return model


def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    preds = predictions.ravel() < 0.5
    return ((preds & labels).sum() +
            (np.logical_not(preds) & np.logical_not(labels)).sum()) / float(labels.size)


# the data, shuffled and split between train and test sets
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
#print(x_test.shape)

response = 'a'

# Will ask the user whether he wants to load or create new matrix
while True:
	print('Press [l] to load matrix or [n] to create new dataset')
	response = raw_input()

	if response == 'l':
		break
	if response == 'n':
		break

begin = time()
if response == 'l':
	matrix_path = '../numpy-matrix/test.npy'
	x_test = np.load(matrix_path)
else:

	x_test  = load_data_test('../dataset/little_test/')


print('Generate / Load time = ', (time()-begin), 's')


expon = x_test.shape[2]*x_test.shape[2]


x_test = x_test.astype('float32')



#input_dim = expon
input_dim = (3, x_test.shape[2], x_test.shape[2])


# network definition
base_network = create_base_network(input_dim)

input_a = Input(shape=(input_dim))
input_b = Input(shape=(input_dim))

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
processed_a = base_network(input_a)
processed_b = base_network(input_b)


distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

model = Model(input=[input_a, input_b], output=distance)


model.load_weights(modelPath)

rms = RMSprop()
model.compile(loss=contrastive_loss, optimizer=rms)


# Load submission set
csvPath = '../csv/submission_info.csv'
df = pd.read_csv(csvPath)

mat = df.as_matrix(df)

csv = open('result.csv','w')

beginStr = 'index,sameArtist\n'
csv.write(beginStr)

count = 0
for i in range(0, mat.shape[0]):
	img1sub = df['img1'][i]
	img2sub = df['img2'][i]
	
	img1str = '../dataset/little_test/little_test/' + img1sub
	img2str = '../dataset/little_test/little_test/' + img2sub

	img1 = Image.open(img1str)
	img2 = Image.open(img2str)
	
	img1 = np.array(img1).transpose()
	img2 = np.array(img2).transpose()

	img1 = np.expand_dims(img1,axis=0)
	img2 = np.expand_dims(img2,axis=0)



	if img1.shape[1] is not 3 or img2.shape[1] is not 3:
		#print '0.2'
		predStr = str(count) + ',' + '0\n'
		csv.write(predStr)

		count = count + 1
		continue

	pred = model.predict([img1, img2])

	if pred[0][0] > 0.5:
		pred = 1
	else:
		pred = 0
	
	predStr = str(count) + ',' + str(pred) +'\n'
	csv.write(predStr)
	
	count = count + 1
###

#pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])

