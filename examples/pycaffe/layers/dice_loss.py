import caffe
import numpy as np
import scipy

class Dice_Loss(caffe.Layer):
    """
    Compute the Euclidean Loss in the same manner as the C++ EuclideanLossLayer
    to demonstrate the class interface for developing layers in Python.
    """

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")
        self.batch_size=bottom[1].data.shape[0]

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count!=bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        self.diff=np.zeros_like(bottom[0].data,dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        self.pd= bottom[0].data
        self.gt = bottom[1].data
        
        # self.diff[...]=bottom[1].data
        smooth = 1.
        self.intersection =  np.sum(self.pd* self.gt)   
        self.sum=np.sum(bottom[0].data)+np.sum(bottom[1].data.sum())+smooth
        self.dice=(2.* self.intersection+smooth)/self.sum
        top[0].data[...] = 1.- self.dice

    def backward(self, top, propagate_down, bottom):
        if propagate_down[1]:
            raise Exception("label not diff")
        elif propagate_down[0]:
            numerator =  2.*np.power(self.gt,2)
            denominator = np.power((np.power(self.pd, 1)+np.power(self.gt,1)),2)
            derivative = numerator/(denominator)/bottom[0].num
            # print(derivative.shape, bottom[0].num)
            bottom[0].diff[...] = derivative