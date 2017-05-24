'''
Created on 22-May-2017

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
import cv2
import lasagne
import matplotlib
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
from lasagne.nonlinearities import softmax


def relu1(x):
    return T.switch(x < 0, 0, x)

from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool

from activ2 import cnn

import sys


#################################################################

def log(f, txt, do_print = 1):
    txt = str(datetime.datetime.now()) + ': ' + txt
    if do_print == 1:
        print(txt)
    f.write(txt + '\n')



#################################################################
def dump_image_batch(X, file_name, max_size = 10, figsize = (10, 10)):
    from itertools import product as iter_product
    nrows = max_size
    ncols = nrows
    if 1 == 1:
        if 1 == 1:
            if 1 == 1:
                shape = X.shape
                figs, axes = plt.subplots(nrows, ncols, figsize = figsize,
                                          squeeze = False)

                for ax in axes.flatten():
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.axis('off')
                for i, (r, c) in enumerate(iter_product(range(nrows), range(ncols))):
                    if i >= shape[0]:
                        break
                    img = X[i].transpose((1, 2, 0))
                    axes[r, c].imshow(img, interpolation = 'none')

                plt.savefig(os.path.join('', file_name))
                #plt.cla()
                plt.clf()
                plt.close()

#################################################################


def test_net(network, dataset):
    loss = 0.0
    accuracy = 0.0
    for i, (X, Y) in enumerate(dataset):
        _, loss_tmp, acc_tmp = network.test(X, Y)
        loss += loss_tmp
        accuracy += acc_tmp
    i += 1
    loss /= i
    accuracy /= i
    txt = 'Accuracy: %.4f%%, loss: %.12f, i: %d' \
            % (accuracy * 100.0, loss, i)
    return loss, accuracy, txt


def train1_net(network, datasets, log_path):
    snapshot_path = os.path.join(log_path, 'snapshots')
    LR = 1.0e-1
    f = open(os.path.join(log_path, 'train.log'), 'w')
    log(f,'6 Layer, Glorut, Without regularization')
    log(f,'Baseline5')
    log(f,'sign activation,pass through gradient' )
    log(f,'gradient cancelled if |x| >1 ')
    log(f, 'Training1...\nTesting before starting...')
    log(f, 'Learning rates LR: %f ' % (LR))

    num_epochs = 60
    losses = np.zeros((2, num_epochs + 1))
    accuracies= np.zeros((2, num_epochs + 1))
    #initial validation loss and accuracy
    losses[1, 0], accuracies[1, 0], txt = test_net(network,
                                                     datasets['valid'])

    log(f, 'TEST valid epoch: ' + str(-1) + ' ' + txt)


    ii = 0
    for epoch in range(num_epochs):
        train_loss = 0.0
        train_accuracy = 0.0
        for i, (X, Y) in enumerate(datasets['train']):
            ii += 1

            loss, accuracy = network.train(X, Y, LR)
            train_loss += loss
            train_accuracy += accuracy
            if ii % 200 == 0:
                log(f, 'Iter: %d [%d], loss: %f, acc: %.2f%%, '
                    'avg_loss: %f, avg_acc: %.2f%%'
                    % (ii, epoch, loss, accuracy, train_loss / (i + 1),
                       100.0 * (train_accuracy / (i + 1))))

        train_loss /= i
        train_accuracy /= i
        losses[0, epoch] = train_loss
        accuracies[0, epoch] = train_accuracy

        log(f, '\nEpoch %d: avg_Loss: %.12f, avg_Acc: %.12f'
            % (epoch, train_loss, train_accuracy * 100.0))

        epoch += 1

        losses[1, epoch], accuracies[1, epoch], txt = test_net(network,
                                                    datasets['valid'])
        log(f, 'TEST valid epoch: ' + str(epoch) + ' ' + txt)

        if (epoch % 20 == 0):#or epoch == 30):
            LR /= 10
            log(f, 'LR CHANGE: %.12f' % (LR))

        p1, = plt.plot(losses[0, : epoch],label='Training loss')
        p2, = plt.plot(losses[1, : epoch],label='Validation loss')
        plt.legend()
        plt.ylabel('LOSS')
        plt.xlabel('EPOCH NUMBER')
        plt.savefig(os.path.join(snapshot_path, 'losses.jpg'))
        plt.clf()
        plt.close()

        p1, = plt.plot(100*accuracies[0, : epoch],label='Training accuracy')
        p2, = plt.plot(100*accuracies[1, : epoch],label='Validation accuracy')
        plt.legend()
	plt.ylabel('ACCURACY %')
        plt.xlabel('EPOCH NUMBER')
        plt.savefig(os.path.join(snapshot_path, 'accuracy.jpg'))
        plt.clf()
        plt.close()


    np.save(os.path.join(snapshot_path, 'losses.npy'), losses)
    '''
    min_epoch = np.argmin(losses)
    log(f, 'Done Training.\n Minimum loss %f at epoch %d' %
        (losses[min_epoch], min_epoch))
    '''
    log(f, '\nTesting at last epoch...')
    _, _, txt = test_net(network, datasets['test'])
    log(f, 'epoch: ' + str(epoch) + ' ' + txt)
    log(f, 'Exiting train...')
    f.close()
    return


#################################################################

#kind of main function
def train():
    train_id = 1
    data_path = '/home/vikram/dataSets'
    #data_path = '../data/cifar-100-python'
    model_save_path = './models'
    batch_size = 128

    print 'Loading Dataset'
    cifar_ds = cifar10_data_set(data_path, batch_size)
    print 'done loading'
    datasets = cifar_ds.data_sets
    print type(datasets['train'].X)
    print np.max(datasets['train'].X), np.min(datasets['train'].X)
    print datasets['train'].X.shape, datasets['train'].Y.shape
    print datasets['test'].X.shape, datasets['test'].Y.shape
    print datasets['valid'].X.shape, datasets['valid'].Y.shape

    X_mean = cifar_ds.X_mean
    # for X, Y in datasets['train']:
    #     dump_image_batch(X, './models/X_train.png')
    #     break
    # for X, Y in datasets['test']:
    #     dump_image_batch(X + X_mean, './models/X_test.png')
    #     break
    # for X, Y in datasets['valid']:
    #     dump_image_batch(X + X_mean, './models/X_valid.png')
    #     break
    #exit(0)

    '''
    cifar_ds.visualize_CIFAR10(datasets['train'].X, datasets['train'].Y)
    cifar_ds.visualize_CIFAR10(datasets['test'].X, datasets['test'].Y)
    cifar_ds.visualize_CIFAR10(datasets['valid'].X, datasets['valid'].Y)
    exit(0)
    '''

    data_shape = datasets['train'].X.shape
    data_shape = (batch_size, ) + data_shape[1: ]
    print 'Data shape:', data_shape

    print 'Creating cnn'
    network = cnn(data_shape)


    if train_id <= 1:
        path = os.path.join(model_save_path, 'train1')
        if not os.path.exists(path):
            os.makedirs(path)
            os.makedirs(os.path.join(path, 'snapshots'))
        train1_net(network, datasets, path)
        f = open(os.path.join(path, network.name + '.save'), 'wb')
        #theano.misc.pkl_utils.dump()
        sys.setrecursionlimit(50000)
        cPickle.dump(network, f, protocol = cPickle.HIGHEST_PROTOCOL)
        f.close()
        print 'Saving weights...'
        network.save_model(os.path.join(path, network.name + ''))
        #network.save_model_np(os.path.join(path, network.name + '_weights'))
        print 'Done'




#################################################################


if __name__ == '__main__':
    train()
