import caffe
import numpy as np
import scipy

class Multi_Loss(caffe.Layer):
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
        self.pd = bottom[0].data
        self.gt = bottom[1].data
        self.diff[...]=bottom[1].data
        smooth = 1.    
        self.sum=bottom[0].data.sum()+bottom[1].data.sum()+smooth
        self.dice=(2.* (bottom[0].data * bottom [1].data).sum()+smooth)/self.sum
        top[0].data[...] = 1.- self.dice

    # def backward(self, top, propagate_down, bottom):
    #     if propagate_down[1]:
    #         raise Exception("label not diff")
    #     elif propagate_down[0]:
    #         bottom[0].diff[...] = -2.*(self.diff**2)/(self.sum**2)
    #     else:
    #         raise Exception("no diff")
    def backward(self, top, propagate_down, bottom):
          if propagate_down[1]:
            raise Exception("label not diff")
          elif propagate_down[0]:
            numerator = self.gt*(np.power(self.pd, 2)+np.power(self.gt,*2).sum()-2*self.pd*(self.gt*self.pd).sum()
            denominator = np.power((np.power(self.pd, 2)+np.power(self.gt,*2)),2)
            bottom[0].diff[...] = -2.*numerator/denominator