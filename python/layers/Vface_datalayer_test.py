import caffe
import numpy as np
from PIL import Image
from pickle import load
import random
import os


class Vface_datalayer(caffe.Layer):
    """
    @Usage: Load a batch of video face and image face pairs as input data.
        The image face is always the first image of one identity, and
        video faces are randomly chosen from the other images of the same identity.
    """
    def setup(self,bottom,top):
        """
        Setup data layer according to parameters
        """
        # config
        self.dataset_root = "/home/wangzd/CASIA-WebFace-5pts-aligned"
        self.image_size = 128
        self.input_size = 32
        self.batch_size = 100
        self.im_fea_dim = 160
        self.mean = (104.00698793, 116.66876762, 122.67891434)#BGR
        self.image_fea_pickle = "/home/wangzd/caffe/JLmodels/im_face.fea"
        
        # four tops: data and same flag
        if len(top) != 4:
            raise Exception("Need to define four tops: video,video_original,image_feature and same flag")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # from the pickle binary load the image face features
        f_im_face_pickle = open(self.image_fea_pickle)
        self.dic_im_face = load(f_im_face_pickle)
        self.k = self.dic_im_face.keys()
        
     
        pass
    def reshape(self,bottom,top):
        top[0].reshape(self.batch_size,3,self.image_size,self.image_size)
        top[1].reshape(self.batch_size,3,self.input_size,self.input_size)
        top[2].reshape(self.batch_size,self.im_fea_dim)
        top[3].reshape(self.batch_size)
        
    def forward(self,bottom,top):

        
        # put data into the top blob
        top[0].data[...] ,top[1].data[...],top[2].data[...],top[3].data[...] \
        =self.fetch_batch(self.batch_size,self.image_size,self.input_size,\
        self.im_fea_dim,self.k,self.dic_im_face,self.dataset_root,self.mean)
        
    def backward(self,top,propagate_down,bottom):
        pass
        

def fetch_batch(batch_size,image_size,input_size,im_fea_dim,k,dic_im_face,dataset_root,mean):
    video_img_original = np.zeros((batch_size,3,image_size,image_size))
    video_img = np.zeros((batch_size,3,input_size,input_size))
    img_fea = np.zeros((batch_size,im_fea_dim))
    same = np.zeros((batch_size))-1
    # fetch the same pairs, half the batchsize
    for i in range((batch_size)/2):
        # randomly choose an identity
        idx = random.randint(0,len(k)-1)
        # read the image feature
        img_fea[i] = dic_im_face[k[idx]][0,0,0]
        # randomly choose another image from the same identity
        id_dir = dataset_root+'/'+ k[idx]
        im_list = os.listdir(id_dir)
        im_list.sort()
        if len(im_list)>1:
            img_name = im_list[random.randint(1,len(im_list)-1)]
        else:
            img_name = im_list[0]
        # fetch the video image
        fetch_name = id_dir+'/'+img_name
        video_img_original[i,:,:,:] = fetch_image(fetch_name,0,input_size,mean)
        video_img[i,:,:,:] = fetch_image(fetch_name,1,input_size,mean)
        
        # same flag
        same[i] = 1
    
    # fetch the different pairs
    for i in range((batch_size)/2,batch_size):
        # randomly choose an identity
        idx = random.randint(0,len(k)-1)
        idx_diff = randint_except(0,idx,len(k)-1)
        # read the image feature
        img_fea[i] = dic_im_face[k[idx]][0,0,0]
        # randomly choose another image from the another identity
        id_dir = dataset_root+'/'+ k[idx_diff]
        im_list = os.listdir(id_dir)
        img_name = im_list[random.randint(0,len(im_list)-1)]
        # fetch the video image
        video_img_original[i,:,:,:] = fetch_image(id_dir+'/'+img_name,0,input_size,mean)
        video_img[i,:,:,:] = fetch_image(id_dir+'/'+img_name,1,input_size,mean)
        # same flag
        same[i] = 0
           
    # shuffle
    shuffle_list = []
    for i in range(batch_size):
        shuffle_list.append([video_img_original[i,:,:,:],\
                            video_img[i,:,:,:],\
                            img_fea[i,:],\
                            same[i]])
    np.random.shuffle(shuffle_list)
    
    for i in range(batch_size):
        video_img_original[i,:,:,:] = shuffle_list[i][0]
        video_img[i,:,:,:] = shuffle_list[i][1]
        img_fea[i,:] = shuffle_list[i][2]
        same[i] = shuffle_list[i][3]
    
    return video_img_original,video_img,img_fea,same
    

def fetch_image(img_path,resize,input_size,mean):
    img = Image.open(img_path)#BGR
    if resize == 1:
        img = img.resize((input_size,input_size))
    in_ = np.array(img,dtype=np.float32)
    in_ -= mean
    in_ = in_.transpose(2,0,1)
    return in_  


def randint_except(start,e,end):
    """
    Randomly select an integer from a range(start,end), except one integer e
    """
    assert(start<=end-1)
    tmp = random.randint(start,end-1)
    if(tmp>=e):
        return tmp+1
    else:
        return tmp

def deprocess_net_image(image):
    image = image.copy()              # don't modify destructively
    image = image[::-1]               # BGR -> RGB
    image = image.transpose(1, 2, 0)  # CHW -> HWC
    image += [123, 117, 104]          # (approximately) undo mean subtraction

    # clamp values in [0, 255]
    image[image < 0], image[image > 255] = 0, 255

    # round and cast from float32 to uint8
    image = np.round(image)
    image = np.require(image, dtype=np.uint8)

    return image        
if __name__ == "__main__":
    import cv2
    dataset_root = "/home/wangzd/CASIA-WebFace-5pts-aligned"
    image_fea_pickle = "/home/wangzd/caffe/JLmodels/im_face.fea"
    f_im_face_pickle = open(image_fea_pickle)
    dic_im_face = load(f_im_face_pickle)
    k = dic_im_face.keys()
    mean = (104.00698793, 116.66876762, 122.67891434)
    a,b,c,d = fetch_batch(100,128,32,160,k,dic_im_face,dataset_root,mean)
    print a.shape
    print b.shape
    print c.shape
    print d.shape
    for i in range(100):
        cv2.imwrite('%d.jpg'%i,deprocess_net_image(a[i]))
        cv2.imwrite('%dre.jpg'%i,deprocess_net_image(b[i]))
    
    

    