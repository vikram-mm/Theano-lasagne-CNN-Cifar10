'''
Created on ***

@author: VAL LAB
'''


import numpy as np
# import cv2
import os
import random
import scipy.io
import copy
import theano

import matplotlib.pyplot as plt
import cPickle as pickle
from numpy import dtype




class data_set():


    def __init__(self, X, Y, batch_size = 1, do_shuffle = False):
        random.seed(11)
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.do_shuffle = do_shuffle
        self.num_samples = X.shape[0]
        assert(self.num_samples == Y.shape[0])
        self.data_item_dimension = X.shape[1: ]


    #def __iter__(self):
    #    return self


    #? Last batch incomplete batch is DROPPED.
    def __iter__(self):
        if self.do_shuffle:
            #print 'SHUFFLED...'
            ##??? FULL DATA NOT USED. UNCOMMENT.
            indices = np.random.permutation(self.num_samples)
            #indices = np.random.permutation(20)
            #print indices
        else:
            indices = np.arange(self.num_samples)

        ##??? FULL DATA NOT USED. UNCOMMENT.
        for i in range(0, self.num_samples, self.batch_size):
        #for i in range(0, 20, self.batch_size):
            if i + self.batch_size > self.num_samples:
                break
            X_orig = self.X[indices[i: i + self.batch_size]]
            X_aug = X_orig

            if self.do_shuffle:
                X_aug = np.empty(X_orig.shape, dtype = np.float32)
                flip_flags = np.random.randint(2, size = X_orig.shape[0])
                padded = np.pad(X_orig, ((0,0), (0,0), (4,4), (4,4)), mode='constant')
                random_cropped = np.zeros(X_orig.shape, dtype = np.float32)
                crops = np.random.random_integers(0, high = 8, size = (X_orig.shape[0], 2))
                for r, flip in enumerate(flip_flags):
                    tmp = padded[r, :, crops[r, 0]: (crops[r, 0] + 32), crops[r, 1]: (crops[r, 1] + 32)]
                    if flip == 1:
                        tmp = tmp[:, :, :: -1]
                    X_aug[r, :, :, :] = tmp

            yield (X_aug, self.Y[indices[i: i + self.batch_size]])



class cifar10_data_set():


    def __init__(self, data_path, batch_size = 1,
                 valid_set_size_percent = 0.1):
        random.seed(11)
        self.data_path = data_path
        X_train, Y_train, X_test, Y_test = self.load_CIFAR10()
        #self.X_mean = np.mean(X_train, axis = 0)
        valid_set_size = np.ceil(X_train.shape[0] * valid_set_size_percent)
        X_train, Y_train, X_valid, Y_valid = \
            self.create_validation_set(X_train, Y_train, valid_set_size)
        self.X_mean = np.mean(X_train, axis = 0)
        X_train -= self.X_mean
        X_valid -= self.X_mean
        X_test -= self.X_mean
        print 'Mean subtracted dataset.'
        self.data_sets = {
                        'train': data_set(X_train, Y_train, batch_size, True),
                        'test': data_set(X_test, Y_test, batch_size),
                        'valid': data_set(X_valid, Y_valid, batch_size)}


    # Data normalized in [0, 1]
    def load_CIFAR_batch(self, file_name):
        with open(file_name, 'r') as f:
            data_dict = pickle.load(f)
            X = data_dict['data']
            Y = data_dict['labels']
            #X = X.reshape((X.shape[0], 32, 32, 3)).transpose(0, 3, 1, 2)
            X = X.reshape(10000, 3, 32, 32)# / 255.0
            X = X.astype(theano.config.floatX) / np.float32(255.0)
            Y = np.array(Y, dtype = 'int32')
        return X, Y


    def load_CIFAR10(self):
        X_arr = []
        Y_arr = []

        #?? FULL DATA NOT LOADED! range SHOULD BE (1, 6).
        for i in range(1, 6):
            f = os.path.join(self.data_path, "data_batch_%d" % (i, ))
            X, Y = self.load_CIFAR_batch(f)
            X_arr.append(X)
            Y_arr.append(Y)

        X_train = np.concatenate(X_arr)
        Y_train = np.concatenate(Y_arr)
        X_test, Y_test = self.load_CIFAR_batch(os.path.join(self.data_path,
                                                            "test_batch"))
        return X_train, Y_train, X_test, Y_test


    def create_validation_set(self, X, Y, valid_set_size):
        return (X[valid_set_size: ], Y[valid_set_size: ],
                X[: valid_set_size], Y[: valid_set_size])


    def visualize_CIFAR10(self, X_train, Y_train, samples_per_class = 10):
        class_names = ['plane', 'car', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
        num_classes = len(class_names)

        for y, cls in enumerate(class_names):
            idxs = np.flatnonzero(Y_train == y)
            idxs = np.random.choice(idxs, samples_per_class, replace=False)
            for i, idx in enumerate(idxs):
                plt_idx = i * num_classes + y + 1
                plt.subplot(samples_per_class, num_classes, plt_idx)
                X = X_train[idx].transpose(1, 2, 0) * 255.0
                plt.imshow(X.astype('uint8'))
                plt.axis('off')
                if i == 0:
                    plt.title(cls)

        plt.show()
