# imports
import json
import time
import pickle
import scipy.misc
import skimage.io
import caffe

import numpy as np
import os.path as osp

from xml.dom import minidom
from random import shuffle
from threading import Thread
from PIL import Image

from tools import SimpleTransformer


class CaffeMultiLabel(caffe.Layer):

    """
    This is a simple synchronous datalayer for training a multilabel model on
    PASCAL.
    """

    def setup(self, bottom, top):

        self.top_names = ['data', 'label1','label2']

        # === Read input parameters ===

        # params is a python dictionary with layer parameters.
        params = eval(self.param_str)

        # Check the parameters for validity.
        # check_params(params)

        # store input as class variables
        self.batch_size = params['batch_size']

        # Create a batch loader to load the images.
        self.batch_loader = BatchLoader(params, None)

        # === reshape tops ===
        # since we use a fixed input image size, we can shape the data layer
        # once. Else, we'd have to do it in the reshape call.
        top[0].reshape(
            self.batch_size, 1, params['im_shape'][0], params['im_shape'][1])
        # Note the 20 channels (because PASCAL has 20 classes.)
        top[1].reshape(
            self.batch_size, 1, params['im_shape'][0], params['im_shape'][1])
        top[2].reshape(
            self.batch_size, 1, params['im_shape'][0], params['im_shape'][1])

        print_info("PascalMultilabelDataLayerSync", params)

    def forward(self, bottom, top):
        """
        Load data.
        """
        for itt in range(self.batch_size):
            # Use the batch loader to load the next image.
            im, im_label1,im_label2 = self.batch_loader.load_next_image()
            # print(im_label1.shape)
            # Add directly to the caffe data layer
            top[0].data[itt, ...] = im
            top[1].data[itt, ...] = im_label1
            top[2].data[itt, ...] = im_label2

    def reshape(self, bottom, top):
        """
        There is no need to reshape the data, since the input is of fixed size
        (rows and columns)
        """
        pass

    def backward(self, top, propagate_down, bottom):
        """
        These layers does not back propagate
        """
        pass


class BatchLoader(object):

    """
    This class abstracts away the loading of images.
    Images can either be loaded singly, or in a batch. The latter is used for
    the asyncronous data layer to preload batches while other processing is
    performed.
    """

    def __init__(self, params, result):
        self.result = result
        self.batch_size = params['batch_size']
        self.data_root = params['data_root']
        self.im_shape = params['im_shape']
        # get list of image indexes.
        data_train = params['raw_data'] + '.txt'
        data_label1 =  params['label1'] + '.txt'
        data_label2 =  params['label2'] + '.txt'

        self.indexlist_train = [line.rstrip('\n') for line in open(
            osp.join(self.data_root, '', data_train))]

        # self.indexlist_label1 = [line.rstrip('\n') for line in open(
        #     osp.join(self.data_root, '', data_label1))]

        # self.indexlist_label2 = [line.rstrip('\n') for line in open(
        #     osp.join(self.data_root, '', data_label2))]

        self._cur = 0  # current image

        print("BatchLoader initialized with {} images".format(
            len(self.indexlist_train)))

    def load_next_image(self):
        """
        Load the next image in a batch.
        """
        # Did we finish an epoch?
        if self._cur == len(self.indexlist_train):
            self._cur = 0
            shuffle(self.indexlist_train)

        # Load an image
        index = self.indexlist_train[self._cur]  # Get the image index
        image_file_name = index + '.png'
        im = np.asarray(Image.open(
            osp.join(self.data_root, 'Raw200', 'Raw200'+image_file_name)))
        # im=im/255
        # print(osp.join(self.data_root, 'Raw200', 'Raw200'+image_file_name))
        #im = scipy.misc.imresize(im, self.im_shape)  # resize
        im = np.array(Image.fromarray(im).resize(self.im_shape))

        # do a simple horizontal flip as data augmentation
        flip = np.random.choice(2)*2-1#-1/1
        im = im[:, ::flip, :]
        if(np.amax(im)>1):
          im=im/255
        # print("im",np.amax(im))
        # Load and prepare ground truth
        im_label1 = np.asarray(Image.open(
            osp.join(self.data_root, 'Soma200Lab', 'Soma200Lab'+image_file_name)))
        im_label1 = np.array(Image.fromarray(im_label1).resize(self.im_shape))    
        im_label1 = im_label1[:, ::flip, :]
        if(np.max(im_label1) > 1):
          im_label1 = im_label1 / 255
          im_label1[im_label1 > 0.5] = 1
          im_label1[im_label1 <= 0.5] = 0
        # print("im_label1",np.amax(im_label1))
        im_label2 = np.asarray(Image.open(
            osp.join(self.data_root, 'Vessel200Lab','Vessel200Lab'+ image_file_name)))
        im_label2 = np.array(Image.fromarray(im_label2).resize(self.im_shape))  
        im_label2 = im_label2[:, ::flip, :]
        if(np.max(im_label2) > 1):
          im_label2 = im_label2 / 255
          im_label2[im_label2 > 0.5] = 1
          im_label2[im_label2 <= 0.5] = 0 
        # print("im_label2",np.amax(im_label2))
        self._cur += 1
        return im, im_label1,im_label2


def check_params(params):
    """
    A utility function to check the parameters for the data layers.
    """
    assert 'split' in params.keys(
    ), 'Params must include split (train, val, or test).'

    required = ['batch_size', 'data_root', 'im_shape']
    for r in required:
        assert r in params.keys(), 'Params must include {}'.format(r)


def print_info(name, params):
    """
    Output some info regarding the class
    """
    print ("{} initialized for split: {}, with bs: {}, im_shape: {}.".format(
        name,
        params['raw_data'],
        params['batch_size'],
        params['im_shape']))
