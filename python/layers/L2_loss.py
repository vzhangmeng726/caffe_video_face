import caffe
import numpy as np

class L2_loss(caffe.Layer):
    def setup(self,bottom,top):
        if len(bottom)!=2:
            raise Exception("Need two inputs to compute distance.")
    def reshape(self,bottom,top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            print 'bottom[0] shape: ',bottom[0].data.shape
            print 'bottom[1] shape: ',bottom[1].data.shape
            raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)
    def forward(self,bottom,top):
        self.diff[...] = sigmoid(bottom[0].data) - sigmoid(bottom[1].data)
        top[0].data[...] = np.sum(self.diff**2) / bottom[0].num /2
        
    def backward(self,top,propagate_down,bottom):
        #for i in range(2):
        #    if not propagate_down[i]:
        #        continue
        if(propagate_down[0]):
            bottom[0].diff[...] =  self.diff / bottom[0].num
            
def sigmoid(x):
    return 1/(1+np.exp(-x/255))