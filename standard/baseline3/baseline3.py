'''
Created on 17-May-2017

@author: vikram@VAL
'''


import numpy as np
import theano
import theano.tensor as T
import timeit
#import pickle
import cPickle
import os
import datetime
# import cv2
import lasagne
import random
import matplotlib
from numpy import dtype
from collections import OrderedDict
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from theano.compile.nanguardmode import NanGuardMode


from data_readers import data_set, cifar10_data_set


from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import GlobalPoolLayer as GapLayer
from lasagne.nonlinearities import softmax, sigmoid
from lasagne.layers import ElemwiseSumLayer
from lasagne.layers import batch_norm
from lasagne.nonlinearities import rectify


def relu1(x):
    return T.switch(x < 0, 0, x)

from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool

#from visualize import plot_conv_weights,

def std_conv_layer(input, num_filters, filter_shape, pad = 'same',
                   nonlinearity = lasagne.nonlinearities.rectify,
                   W = None,
                   #W = lasagne.init.Normal(std = 0.01, mean = 0.0),
                   b = lasagne.init.Constant(0.),
                   do_batch_norm = False):
    if W == None:
        if nonlinearity == lasagne.nonlinearities.rectify:
            print 'convlayer: rectifier func'
            W = lasagne.init.HeNormal(gain = 'relu')
        else:
            print 'convlayer: sigmoid func'
            W = lasagne.init.HeNormal(1.0)
    else:
        print 'convlayer: W not None'
    conv_layer = ConvLayer(input, num_filters, filter_shape,
                        pad = pad, flip_filters = False,
                        W = W, b = b,
                        nonlinearity = nonlinearity)
    if do_batch_norm:
        conv_layer = lasagne.layers.batch_norm(conv_layer)
    else:
        print 'convlayer: No batch norm.'
    return conv_layer


from lasagne import layers

try:
    from lasagne.layers import TransposedConv2DLayer as DeconvLayer
except:
    from new_conv import TransposedConv2DLayer as DeconvLayer


try:
    from lasagne.layers import ExpressionLayer
except:
    from new_special import ExpressionLayer


from itertools import product as iter_product


###############################################################################



####################################################

def plot_weights(W, plot_name, file_path = '.',
                    max_subplots = 100, max_figures = 32,
                    figsize = (6, 6)):
    W = W.get_value(borrow = True)
    shape = W.shape
    assert((len(shape) == 2) or (len(shape) == 4))
    max_val = np.max(W)
    min_val = np.min(W)

    if len(shape) == 2:
        plt.figure(figsize = figsize)
        plt.imshow(W, cmap = 'gray',#'jet',
                   vmax = max_val, vmin = min_val,
                   interpolation = 'none')
        plt.axis('off')
        plt.colorbar()
        file_name = plot_name + '.png'
        plt.savefig(os.path.join(file_path, file_name))
        plt.clf()
        plt.close()
        return

    nrows = min(np.ceil(np.sqrt(shape[1])).astype(int),
                np.floor(np.sqrt(max_subplots)).astype(int))
    ncols = nrows
    '''
    max_val = -np.inf
    min_val = np.inf
    for i in range(shape[0]):
        tmp = np.mean(W[i], axis = 0)
        max_val = max(max_val, np.max(tmp))
        min_val = min(min_val, np.min(tmp))
    '''
    for j in range(min(shape[0], max_figures)):
        figs, axes = plt.subplots(nrows, ncols, figsize = figsize,
                                  squeeze = False)
        for ax in axes.flatten():
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis('off')
        for i, (r, c) in enumerate(iter_product(range(nrows), range(ncols))):
            if i >= shape[1]:
                break
            im = axes[r, c].imshow(W[j, i], cmap = 'gray',#'jet',
                                   vmax = max_val, vmin = min_val,
                                   interpolation = 'none')
        figs.colorbar(im, ax = axes.ravel().tolist())
        file_name = plot_name + '_fmap' + str(j) + '.png'
        plt.savefig(os.path.join(file_path, file_name))
        plt.clf()
        plt.close()
    return
#################################################################

class cnn():

    def __init__(self, input_shape, input = None,
                 num_class = 10,
                 name = 'baseline1'):
        if input is None:
            input = T.tensor4()
        self.input = input
        self.name = name
        self.num_class = num_class

        print ('Building cnn '+name)
        self.input_layer = InputLayer(shape = input_shape, input_var = self.input)
        self.output_layer=self.build_cnn(self.input_layer)
        momentum = 0.9
        weight_decay = 0.0001
        print('Building theano functions...')


        self.params = lasagne.layers.get_all_params(self.output_layer,
                                                           trainable = True)

        self.train_output = lasagne.layers.get_output(self.output_layer)
        self.test_output = self.train_output

        Y = T.ivector()
        w_reg_l2 = weight_decay * \
                    lasagne.regularization.regularize_network_params(\
                    self.output_layer, lasagne.regularization.l2)
        train_error = lasagne.objectives.\
                        categorical_crossentropy(self.train_output, Y).mean()
        train_loss = train_error #+ w_reg_l2
        train_accuracy = T.mean(T.eq(T.argmax(self.train_output, axis = 1), Y),
                                dtype = theano.config.floatX)
        test_loss = lasagne.objectives.\
                        categorical_crossentropy(self.test_output, Y).mean()
        test_accuracy = T.mean(T.eq(T.argmax(self.test_output, axis = 1), Y),
                                dtype = theano.config.floatX)

        self.test_func = theano.function(
                                inputs = [self.input, Y],
                                outputs = [test_loss, test_accuracy,
                                           self.test_output])

        LR= T.scalar()
        updates = lasagne.updates.momentum(train_loss, self.params,
                                           learning_rate = LR,
                                           momentum = momentum)
        self.train_func = theano.function(
                                inputs = [self.input, Y, LR],
                                outputs = [train_loss, train_accuracy],
                                updates = updates)

        print 'Done.'


    # BEWARE: std_conv_layer may have batch norm.
    def create_net(self, layer_details, prev_layer):
        print self.name, '.create_net> building net...'
        layers_list = [prev_layer]
        #layers_dict = {'input': prev_layer}
        for attributes in layer_details:
            name, layer_fn = attributes[: 2]
            params = []
            params_dict = {}
            if len(attributes) >= 4:
                params, params_dict = attributes[2: 4]
            elif len(attributes) >= 3:
                params = attributes[2]
            print 'layer: ', name
            prev_layer = layer_fn(prev_layer, *params, **params_dict)
            #layers_dict[name] = prev_layer
            layers_list.append(prev_layer)
        print 'done.'
        return layers_list


    def build_cnn(self, input_layer, n=5):



        # Building the network

        layers =[
                    ('conv1', std_conv_layer, [64, 5]),
                    ('pool1', PoolLayer, [2]),
                    ('conv2', std_conv_layer, [128, 3]),
                    ('conv22', std_conv_layer, [128, 3]),
                    ('pool2', PoolLayer, [2]),
                    ('conv3', std_conv_layer, [256, 3]),
                    ('conv32', std_conv_layer, [256, 3]),
                    ('conv33', std_conv_layer, [256, 3]),
                    ('gap1', GapLayer),
                    ('softmax', DenseLayer, [], {'num_units': 10,
                     'W': lasagne.init.HeNormal(1.0),
                     'nonlinearity': lasagne.nonlinearities.softmax})
                ]

        network = self.create_net(layers, input_layer)
        return network[-1]

    def test(self, X, Y):
        loss, accuracy, confidences = self.test_func(X, Y)
        return confidences, loss, accuracy

    def train(self, X, Y, LR):
        loss, accuracy = self.train_func(X, Y, LR)
        return loss, accuracy


    def plot_all_weights(self, dump_path, figsize = (8, 8)):
        for name, net in self.net.iteritems():
            if isinstance(net, react_block):
                plot_weights(name, dump_path)


    def save_model(self, file_name = None, layer_to_save = None):
        if layer_to_save is None:
            layer_to_save = self.output_layer

        print 'Saving model starting from layer', layer_to_save.name, '...'
        print 'filename', file_name
        params = lasagne.layers.get_all_param_values(layer_to_save)
        if file_name is not None:
            fp = open(file_name + '.save', 'wb')
            cPickle.dump(params, fp, protocol = cPickle.HIGHEST_PROTOCOL)
            fp.close()
        print 'Done.'
        return params

    def load_model(self, file_name, layer_to_load = None):
        if layer_to_load is None:
            layer_to_load = self.output_layer
        print 'Loading model starting from layer', layer_to_load.name, '...'
        fp = open(file_name, 'rb')
        params = cPickle.load(fp)
        fp.close()
        lasagne.layers.set_all_param_values(layer_to_load, params)
        print 'Done'

    def load_model_np(self, file_name, layer_to_load = None):
        if layer_to_load is None:
            layer_to_load = self.output_layer
        print 'Loading model starting from layer', layer_to_load.name, '...'
        # load network weights from model file
        with np.load(file_name) as f:
             params = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(layer_to_load, params)
        print 'Done'
