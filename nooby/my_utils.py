'''
Created on 16-May-2017

@author: vikram@VAL
'''
import numpy as np
import theano
import theano.tensor as T
import timeit
import time
import cPickle
import os
import datetime
import cv2
import lasagne
import random
import matplotlib
from numpy import dtype
from collections import OrderedDict
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from theano.compile.nanguardmode import NanGuardMode


from data_reader import data_set


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

#########################################################################

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

#########################################################################

def train(train_fn,val_fn,
            model,
            LR_start,LR_decay,LR_decay_after_num_epochs,
            num_epochs,
            dataset,graph_name,
            save_path=None,
            shuffle_parts=1):

    X_val=dataset.data_sets['valid'].X
    y_val=dataset.data_sets['valid'].Y

    # This function trains the model a full epoch (on the whole dataset)
    def train_epoch():

        train_loss = 0.0
        train_acc = 0.0


        ii=0
        for i, (X, Y) in enumerate(dataset.data_sets['train']):

            ii += 1
            loss = train_fn(X, Y,LR)
            train_loss += loss
            _,acc=val_fn(X,Y)
            train_acc+=acc
            print('Iter: %d [%d], loss: %f, acc: %.2f%%, '
                'avg_loss: %f, avg_acc: %.2f%%'
                % (ii, epoch+1, loss, acc*100, train_loss / (i + 1),
                   (train_acc*100/ (i + 1))))
        i+=1
        train_loss /= i
        train_acc /= i

        return train_acc*100,train_loss


    # This function tests the model a full epoch (on the whole dataset)
    def val_epoch(tag):

        net_loss = 0.0
        net_acc = 0.0


        for i, (X, Y) in enumerate(dataset.data_sets[tag]):


            loss,acc=val_fn(X,Y)
            net_loss+=loss
            net_acc+=acc

        i+=1
        net_loss /= i
        net_acc /= i

        return net_acc*100,net_loss

    best_val_acc=0
    best_epoch = 1
    LR = LR_start

    graphX=[] #stores epoch numbers
    graphY1=[] #stores val errors
    graphY2=[] #stores train errors

    graphYLossTrain=[]
    graphYLossVal=[]

    train_acc=0
    # We iterate over epochs:
    print("Beginning training ... ")

    ii = 0

    for epoch in range(num_epochs):

        start_time = time.time()
        print("epoch : "+str(epoch+1))
        train_acc,train_loss=train_epoch()
        print('Done train epoch '+str(epoch+1))


        val_acc, val_loss = val_epoch(tag='valid')
        print('Done validation epoch')

        # test if validation error went down
        if val_acc >= best_val_acc:

            best_val_acc = val_acc
            best_epoch = epoch+1
            #print('Starting test epoch')
            if save_path is not None:
                np.savez(save_path, *lasagne.layers.get_all_param_values(model))

        epoch_duration = time.time() - start_time

        # Then we print the results for this epoch:
        print("Epoch "+str(epoch + 1)+" of "+str(num_epochs)+" took "+str(epoch_duration)+"s")
        print("  LR:                            "+str(LR))
        print("  training loss:                 "+str(train_loss))
        print("  training accuracy:                 "+str(train_acc))
        print("  validation loss:               "+str(val_loss))
        print("  validation accuracy:         "+str(val_acc)+"%")
        print("  best epoch:                    "+str(best_epoch))
        print("  best validation accuracy:    "+str(best_val_acc)+"%")

        graphX.append(epoch+1)
        graphY1.append(val_acc)
        graphY2.append(train_acc)
	graphYLossTrain.append(train_loss)
        graphYLossVal.append(val_loss)

        plt.plot(graphX,graphY1,label='Validation accuracy')
        plt.plot(graphX,graphY2,label='Train accuracy')
        plt.legend(loc=4)
        plt.ylabel('ACCURACY %')
        plt.xlabel('EPOCH NUMBER')
        #plt.show()
        plt.savefig(graph_name)
        plt.clf()
        plt.close()
	
	plt.plot(graphX,graphYLossVal,label='Validation loss')
        plt.plot(graphX,graphYLossTrain,label='Train loss')
        plt.legend(loc=4)
        plt.ylabel('LOSS')
        plt.xlabel('EPOCH NUMBER')
        #plt.show()
        plt.savefig(graph_name+"_loss")
        plt.clf()
        plt.close()

        if((epoch+1)%LR_decay_after_num_epochs==0):
            LR*=LR_decay
            print('**********Decaying LR, new LR : ' + str(LR)+'************')

    print('***************Training Done*******************')

    test_acc, test_loss = val_epoch('test')
    print("  test loss:                     "+str(test_loss))
    print("  test accuracy:               "+str(test_acc)+"%")
    print("  best validation accuracy:    "+str(best_val_acc)+"%")

    print('*************************************************')

    #final graph
    plt.plot(graphX,graphY1,label='Validation accuracy')
    plt.plot(graphX,graphY2,label='Train accuracy ( calculaed every 5 epochs)')
    plt.legend(loc=4)
    plt.ylabel('ACCURACY %')
    plt.xlabel('EPOCH NUMBER')
    plt.suptitle(" test accuracy:"+str(test_acc)+"\n" "best validation accuracy:  "+str(best_val_acc)+"\n" +"LR_start : "+str(LR_start)+"\n"+"LR_decay : " +str(LR_decay)+"\n"+"LR_decay_after_num_epochs : "+str(LR_decay_after_num_epochs))
    #plt.show()
    plt.savefig(graph_name)
    plt.clf()
    plt.close()
