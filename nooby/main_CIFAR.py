'''
Created on 16-May-2017

@author: vikram@VAL
'''

import theano
import theano.tensor as T
import lasagne
import data_reader
from my_utils import std_conv_layer
from my_utils import train

#########################################################
#Setting the Parameters

batch_size = 128
print("batch_size = "+str(batch_size))

# Training parameters
num_epochs = 60
print("num_epochs = "+str(num_epochs))

# Decaying LR
LR_start = 0.1
print("LR_start = "+str(LR_start))
LR_decay = 0.1
print("LR_decay = "+str(LR_decay))
LR_decay_after_num_epochs=20
print("LR_decay_after_num_epochs = "+str(LR_decay_after_num_epochs))

momentum = 0.9
print('momentum : '+str(momentum))


######################################################
#Loading the data_set
data=data_reader.cifar10_data_set('/home/vikram/dataSets',batch_size=batch_size)

###########################################################

print("building the cnn...")

# Prepare Theano variables for inputs and targets
input = T.tensor4('inputs')
target = T.ivector('targets')
LR = T.scalar('LR', dtype=theano.config.floatX)

cnn = lasagne.layers.InputLayer(
        shape=(None, 3, 32, 32),
        input_var=input)
cnn=std_conv_layer(cnn,64,5)
cnn=cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))
cnn=std_conv_layer(cnn,128,3)
cnn=std_conv_layer(cnn,128,3)
cnn=cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))
cnn=std_conv_layer(cnn,256,3)
cnn=std_conv_layer(cnn,256,3)
cnn=std_conv_layer(cnn,256,3)
cnn=lasagne.layers.GlobalPoolLayer(cnn)
cnn = lasagne.layers.DenseLayer(cnn,num_units=10,
                                    W=lasagne.init.HeNormal(1.0),
                                    nonlinearity=lasagne.nonlinearities.softmax)

###########################################################
#Set up the theano functions

train_output=lasagne.layers.get_output(cnn)

loss = lasagne.objectives.categorical_crossentropy(train_output, target)
loss = loss.mean()

params = lasagne.layers.get_all_params(cnn, trainable=True)

updates = lasagne.updates.momentum(loss_or_grads=loss, params=params, learning_rate=LR,momentum=momentum)

train_fn = theano.function([input, target,LR], loss, updates=updates)



test_prediction = lasagne.layers.get_output(cnn, deterministic=True)

test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                        target)
test_loss = test_loss.mean()

test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target),
                  dtype=theano.config.floatX)

val_fn = theano.function([input, target], [test_loss, test_acc])

#################################################################
print('CNN Built , calling train function')
#finally the train function

train(
        train_fn,val_fn,
        cnn,
        LR_start,LR_decay,LR_decay_after_num_epochs,
        num_epochs,
        data,
        save_path='/home/vikram/exp16May_Final/',
        graph_name='cifar_ReLuActivation',
        shuffle_parts=1)
######################################################################
