from __future__ import absolute_import
from __future__ import print_function
import numpy as np

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

from imageDataExtract import *


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


def create_pairs(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []

    print(len(digit_indices))

    rangeNum = 11

    n = min([len(digit_indices[d]) for d in range(rangeNum)]) - 1
    for d in range(rangeNum):
        for i in range(n):

	    print(digit_indices[d].shape)

            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, 10)
            dn = (d + inc) % 10
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)


def create_base_network():
	'''Base network to be shared (eq. to feature extraction).
	'''
	'''
	seq = Sequential()
	seq.add(Dense(128, input_shape=(input_dim,), activation='relu'))
	seq.add(Dropout(0.1))
	seq.add(Dense(128, activation='relu'))
	seq.add(Dropout(0.1))
	seq.add(Dense(128, activation='relu'))
	'''


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
	model.add(Dense(64, activation='relu'))

    
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
	matrix_path = '../numpy-matrix-nongrayscale/main.npy'
	label_path = '../numpy-matrix-nongrayscale/label.npy'
	x_train, y_train, x_test, y_test = load_matrix(matrix_path, label_path)
else:

	x_train, x_test, y_train, y_test  = load_data('../dataset/little_train/')
print('Generate / Load time = ', (time()-begin), 's')

expon = x_train.shape[2]*x_train.shape[2]
'''
x_train = x_train.reshape(x_train.shape[0], expon)
x_test = x_test.reshape(x_test.shape[0], expon)
'''
#print(x_test.shape)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


print(x_train.shape)

input_dim = (3, x_train.shape[2], x_train.shape[2])
nb_epoch = 20



# create training+test positive and negative pairs
digit_indices = [np.where(y_train == i)[0] for i in y_train]
tr_pairs, tr_y = create_pairs(x_train, digit_indices)

digit_indices = [np.where(y_test == i)[0] for i in y_test]
te_pairs, te_y = create_pairs(x_test, digit_indices)
print(te_y.shape)

print('------')
print(len(digit_indices))

# network definition
base_network = create_base_network()

input_a = Input(shape=(input_dim))
input_b = Input(shape=(input_dim))

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
processed_a = base_network(input_a)
processed_b = base_network(input_b)


distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

model = Model(input=[input_a, input_b], output=distance)
print('Shape:')
print(tr_pairs.shape)
print(te_pairs.shape)

# train
rms = RMSprop()
model.compile(loss=contrastive_loss, optimizer=rms)
model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
                  validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y),
                  batch_size=128,
                  nb_epoch=nb_epoch)



# compute final accuracy on training and test sets
pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
tr_acc = compute_accuracy(pred, tr_y)
pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
te_acc = compute_accuracy(pred, te_y)

print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))

# serialize weights to HDF5
model.save_weights("../models/model.h5")
print('')
print("Saved model to disk")
