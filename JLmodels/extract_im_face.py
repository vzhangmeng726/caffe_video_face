caffe_root = '/home/wangzd/caffe/'
webface_root = '/home/wangzd/CASIA-WebFace-5pts-aligned'
batch_size = 50
input_scale = 128


import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
import os
import numpy as np
from pickle import dump
from pickle import load



# Some useful functions
def extract_feature(img,net,layer_name):
    '''
    @usage: Extract features using the input net)
    @params:
        img: A list of images which is transformed to BGR, with shape N*C*H*W,
             where N is batch size, C is chanel number, H and W are heights and
             weights.(Note that this version the input is just one image not a
             list. This will be modified later.)
        net: The net used for extracting features.
        layer_name: The layer's name from which you want extract feature.
    @output:
        A list of vectors.(Note that this version the input is just one vector,
        not a list)
    '''
    net.blobs['data'].data[...] = img
    output = net.forward()
    feature = output[layer_name][0]
    return feature

def transpose_image(img):
    c_1 = img[1,:,:].T
    c_2 = img[2,:,:].T
    c_0 = img[0,:,:].T
    im = np.zeros_like(img)
    im[1,:,:]= c_1
    im[2,:,:]= c_2
    im[0,:,:]= c_0
    return im

# Reading config
model_weights = caffe_root + 'JLmodels/models/facerec_iter_162000.caffemodel'
if(os.path.isfile(model_weights)):
    print('Model weights found!')
else:
    print('Model weights not found!')
model_def =caffe_root + 'JLmodels/facerec_deploy.prototxt'
if(os.path.isfile(model_def)):
    print('Model defination found!')
else:
    print('Model defination not found!')

# Initialization
caffe.set_device(0)
caffe.set_mode_gpu()
net = caffe.Net(model_def, model_weights, caffe.TEST)

# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
print 'mean-subtracted values:', zip('BGR', mu)

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
#transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

# set the size of the input (we can skip this if we're happy
#  with the default; we can also change it later, e.g., for different batch sizes)
net.blobs['data'].reshape(batch_size,        # batch size
                          3,         # 3-channel (BGR) images
                          input_scale, input_scale)  # image size is 227x227

id_list = os.listdir(webface_root)
id_list.sort()

fea_dic ={}
count = 0
for id in id_list:
    count += 1
    im_list = os.listdir(webface_root+'/'+id)
    im_list.sort()
    if(len(id_list)==0):
        fea = -1
    else:
        im_path = webface_root+'/'+id+'/'+im_list[0]
        im = caffe.io.load_image(im_path)
        im = transformer.preprocess('data',im)
        fea = extract_feature(im,net,'l2').copy()
    fea_dic[id] = fea
    if(count%500==0):
        print 'Extracting: ',count,'/',len(id_list)
        
save_file = open('im_face.fea','w')
dump(fea_dic,save_file)
save_file.close()
    
    
    
    
    
    
    
    
    
    
                          
                          