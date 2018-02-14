# -*- coding: utf-8 -*-
'''
modify from https://github.com/keras-team/keras/blob/master/examples/image_ocr.py

try to make thing simple to know how Keras API works
'''

# source: https://github.com/keras-team/keras/blob/master/examples/image_ocr.py

import os
import itertools
import codecs
import re
import datetime
import editdistance
import numpy as np
from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation
from keras.layers import Reshape, Lambda
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.layers.recurrent import GRU
from keras.optimizers import SGD
from keras.utils.data_utils import get_file
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint, CSVLogger
import keras.callbacks
import json
import random
from . import paint_text as _paint_text
import time
import codelog_image_ocr.c as c
import string
from . import model as my_model
import argparse
from . import g

# character classes and matching regex filter
char_set = string.digits

np.random.seed(55)

paint_text = _paint_text.paint_text

# Translation of characters to unique integer values
def text_to_labels(text):
    ret = []
    for char in text:
        ret.append(char_set.find(char))
    return ret


# Reverse translation of numerical classes back to characters
def labels_to_text(labels):
    ret = []
    for c in labels:
        if c == len(char_set):  # CTC Blank
            ret.append("")
        else:
            ret.append(char_set[c])
    return "".join(ret)


def random_string(max_size,char_list):
    size = random.randint(1,max_size)
    return ''.join(random.choice(char_list)for _ in range(size))


# Uses generator functions to supply train/test with
# data. Image renderings are text are created on the fly
# each time with random perturbations

class TextImageGenerator():

    def __init__(self, minibatch_size,
                 img_w, img_h, downsample_factor,
                 absolute_max_string_len=16):

        self.minibatch_size = minibatch_size
        self.img_w = img_w
        self.img_h = img_h
        self.downsample_factor = downsample_factor
        self.blank_label = self.get_output_size() - 1
        self.absolute_max_string_len = absolute_max_string_len

    def get_output_size(self):
        return len(char_set) + 1

    # each time an image is requested from train/val/test, a new random
    # painting of the text is performed
    # size:  self.minibatch_size
    def get_batch(self, size):
        # print('get_batch size={}'.format(size))
        # width and height are backwards from typical Keras convention
        # because width is the time dimension when it gets fed into the RNN
        if K.image_data_format() == 'channels_first':
            X_data = np.ones([size, 1, self.img_w, self.img_h])
        else:
            X_data = np.ones([size, self.img_w, self.img_h, 1])

        labels = np.ones([size, self.absolute_max_string_len])
        input_length = np.zeros([size, 1])
        label_length = np.zeros([size, 1])
        source_str = []
        for i in range(size):
            word = random_string(4,char_set)
            if K.image_data_format() == 'channels_first':
                X_data[i, 0, 0:self.img_w, :] = self.paint_func(word)[0, :, :].T
            else:
                X_data[i, 0:self.img_w, :, 0] = self.paint_func(word)[0, :, :].T
            labels[i, 0:len(word)] = text_to_labels(word)
            input_length[i] = self.img_w // self.downsample_factor - 2
            label_length[i] = len(word)
            source_str.append(word)
        inputs = {'the_input': X_data,
                  'the_labels': labels,
                  'input_length': input_length,
                  'label_length': label_length,
                  'source_str': source_str  # used for visualization only
                  }
        outputs = {'ctc': np.zeros([size])}  # dummy data for dummy loss function
        return (inputs, outputs)

    def next_batch(self):
        while 1:
            ret = self.get_batch(self.minibatch_size)
            yield ret

    def paint_func(self, text):
        return paint_text(text, self.img_w, self.img_h, rotate=False, ud=False, multi_fonts=False)


# the actual loss calc occurs here despite it not being
# an internal Keras loss function

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


# For a real OCR application, this should be beam search with a dictionary
# and language model.  For this example, best path is sufficient.

def decode_batch(test_func, word_batch):
    out = test_func([word_batch])[0]
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = labels_to_text(out_best)
        ret.append(outstr)
    return ret


def train(epochs, img_w, output, gs_output, steps_per_epoch, validation_steps, minibatch_size, **kwargs):
    # Input Parameters
    img_h = 64
    output_dir = output
    
    c.reset_dir(output_dir)

    # Network parameters
    pool_size = my_model.pool_size

    img_gen = TextImageGenerator(
        minibatch_size=minibatch_size,
        img_w=img_w,
        img_h=img_h,
        downsample_factor=(pool_size ** 2)
    )

    input_data, y_pred = my_model.create_tensor_io(img_w, img_h, img_gen.get_output_size())

    labels = Input(name='the_labels', shape=[img_gen.absolute_max_string_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    # clipnorm seems to speeds up convergence
    sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)
    # captures output of softmax so we can decode the output during visualization
    test_func = K.function([input_data], [y_pred])

    callbacks = []
    callbacks.append(ModelCheckpoint(filepath=os.path.join(output_dir,'weight.{epoch:06d}.hdf5')))
    callbacks.append(ModelCheckpoint(filepath=os.path.join(output_dir,'weight.best.hdf5'), save_best_only=True))
    callbacks.append(CSVLogger(filename=os.path.join(output_dir,'log.csv')))
    if gs_output != None:
        callbacks.append(g.Copy(os.path.join(output_dir,'weight.{epoch:06d}.hdf5'),os.path.join(gs_output,'weight.{epoch:06d}.hdf5')))
        callbacks.append(g.Copy(os.path.join(output_dir,'weight.best.hdf5'),       os.path.join(gs_output,'weight.best.hdf5')))
        callbacks.append(g.Copy(os.path.join(output_dir,'log.csv'),                os.path.join(gs_output,'log.csv')))

    model.fit_generator(generator=img_gen.next_batch(),
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs,
                        validation_data=img_gen.next_batch(),
                        validation_steps=validation_steps,
                        callbacks=callbacks,
                        verbose=2
                        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--job-dir', type=str, default='output', help='output dir')
    parser.add_argument('--epochs', type=int, default=10, help='epochs')
    parser.add_argument('--img-w', type=int, default=128, help='img-w')
    parser.add_argument('--steps_per_epoch', type=int, default=10, help='steps_per_epoch')
    parser.add_argument('--validation_steps', type=int, default=10, help='validation_steps')
    parser.add_argument('--minibatch_size', type=int, default=32, help='minibatch_size')
    
    parse_args = parser.parse_args()

    arg_dict = dict(parse_args.__dict__)
    
    if arg_dict['job_dir'].startswith('gs://'):
        arg_dict['gs_output']  = arg_dict['job_dir']
        arg_dict['output']     = 'gs'
    else:
        arg_dict['gs_output']  = None
        arg_dict['output']     = arg_dict['job_dir']
    
    train(**arg_dict)
