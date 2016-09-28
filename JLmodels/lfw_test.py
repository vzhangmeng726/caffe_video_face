# Settings
caffe_root = '/home/wangzd/caffe/'
lfw_root = '/home/wangzd/lfw-hallucination-aligned'
pairs_path = 'pairs.txt'
roc_txt = 'roc_H1.txt'
batch_size = 50
input_scale = 128


import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
import os
import numpy as np
import matplotlib.pyplot as plt
from pickle import dump
from pickle import load


# set display defaults
plt.rcParams['figure.figsize'] = (10, 10)        # large images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap


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

def get_cos_dist(fea_1,fea_2):
    fea_1 =  np.squeeze(fea_1)
    fea_2 =  np.squeeze(fea_2)
    return np.dot(fea_1.T,fea_2)/(np.linalg.norm(fea_2)*np.linalg.norm(fea_1))

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

# Read the LFW pair list
is_same      = []
image_list_1 = []
image_list_2 = []


pair_txt = open(pairs_path,'r')                     
for line in pair_txt.readlines():
    info = line.strip().split('\t')
    if(len(info)==3):
        is_same.append(True)
        index_1 = int(info[1])
        index_2 = int(info[2])
        image_list_1.append(lfw_root +'/'+ info[0] + '/' + info[0] + '_' + (('%04d.jpg')% (index_1)))
        image_list_2.append(lfw_root +'/'+ info[0] + '/' + info[0] + '_' + (('%04d.jpg')% (index_2)))
    if(len(info)==4):
        is_same.append(False)
        index_1 = int(info[1])
        index_2 = int(info[3])                  
        image_list_1.append(lfw_root +'/'+ info[0] + '/' + info[0] + '_' + (('%04d.jpg')% (index_1)))
        image_list_2.append(lfw_root +'/'+ info[2] + '/' + info[2] + '_' + (('%04d.jpg')% (index_2)))


pair_txt.close()


        
# Main Process
num_pairs = len(is_same) # This number should be 6000
fea_1 = []
fea_2 = []
# Ectracting features
#for i in range(num_pairs):
#    img_1 = caffe.io.load_image(image_list_1[i])
#    img_1 = transformer.preprocess('data',img_1)
#    #img_1 = transpose_image(img_1)
#    fea_1.append(extract_feature(img_1,net,'l2').copy())
#    img_2 = caffe.io.load_image(image_list_2[i])
#    img_2 = transformer.preprocess('data',img_2)
#    #img_2 = transpose_image(img_2)
#    fea_2.append(extract_feature(img_2,net,'l2').copy())
#    if(i%100==0):
#        print 'Extracting: ',i,'/',num_pairs

#save_file_1 = open('features_1_lfw_H1','w')
#save_file_2 = open('features_2_lfw_H1','w')
#dump(fea_1,save_file_1)
#dump(fea_2,save_file_2)
#save_file_1.close()
#save_file_2.close()

# Load the features
f_1 = open('features_1_lfw_H1')
f_2 = open('features_2_lfw_H1')
fea_1 = load(f_1)
fea_2 = load(f_2)
f_1.close()
f_2.close()

threshold = 0
print '===========================START GENERATING ROC=========================='
roc_file = open(roc_txt,'w+')
while True:
    if(threshold>1.50):
        break
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    for i in range(num_pairs):
        same = is_same[i]
        dist = np.linalg.norm(fea_1[i]-fea_2[i])
        if(same and dist<threshold):
            TP += 1
        if(same and dist>threshold):
            FN += 1
        if((not same) and dist>threshold):
            TN += 1
        if((not same) and dist<threshold):
            FP += 1
    print 'Threshold: ', threshold, '\tTPR: ',float(TP)/(TP+FN),'\tFPR: ',float(FP)/(FP+TN), '\tACC: ',float((TP+TN))/num_pairs
    out = str(float(TP)/(TP+FN))+ '\t'+str(float(FP)/(FP+TN))+'\n'
    roc_file.write(out)

    threshold += 0.002

print '===========================START CROSS VALIDATION=========================='
num_fold = 600
acc = []
for fold in range(10):
    val_start_idx = fold*num_fold
    thres = 0
    thres_list =[]
    acc_list = []
    while True:
        if(thres>1.50):
            break
        TP = 0
        TN = 0
        FN = 0
        FP = 0
        for i in range(num_pairs):
            if(i>=val_start_idx and i<val_start_idx+num_fold):
                continue
            else:
                same = is_same[i]
                dist = np.linalg.norm(fea_1[i]-fea_2[i])
                if(same and dist<thres):
                    TP += 1
                if(same and dist>thres):
                    FN += 1
                if((not same) and dist>thres):
                    TN += 1
                if((not same) and dist<thres):
                    FP += 1
        acc_list.append(float((TP+TN))/(num_pairs-num_fold))
        thres_list.append(thres)
        thres += 0.002
    thres_this_fold = thres_list[np.argmax(acc_list)]
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    for i in range(val_start_idx,val_start_idx+num_fold):
        same = is_same[i]
        dist = np.linalg.norm(fea_1[i]-fea_2[i])
        if(same and dist<thres_this_fold):
            TP += 1
        if(same and dist>thres_this_fold):
            FN += 1
        if((not same) and dist>thres_this_fold):
            TN += 1
        if((not same) and dist<thres_this_fold):
            FP += 1
    acc.append(float(TP+TN)/num_fold)
    print 'Fold: ',fold, '\tACC: ', acc[-1]
print 'Done. Mean ACC: ', np.mean(acc), '\tVariance: ', np.var(acc)  
        
        
        
        
        
        
        
        
        
        
        


        
