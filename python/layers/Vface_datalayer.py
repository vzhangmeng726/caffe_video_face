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
        self.mean = (104.00698793, 116.66876762, 122.67891434)
        self.image_fea_pickle = "/home/wangzd/caffe/JLmodels/im_face.fea"
        
        # two tops: data and same flag
        #if len(top) != 4:
        #    raise Exception("Need to define two tops: video,image_feature and same flag")
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
        #top[2].reshape(self.batch_size,self.im_fea_dim)
        #top[3].reshape(self.batch_size)
        
    def forward(self,bottom,top):
        list_id_im =[]
        list_id_vd =[]
        video_img_original = np.zeros((self.batch_size,3,self.image_size,self.image_size))
        video_img = np.zeros((self.batch_size,3,self.input_size,self.input_size))
        img_fea = np.zeros((self.batch_size,self.im_fea_dim))
        same = np.zeros((self.batch_size))-1
        # fetch the same pairs, half the batchsize
        for i in range((self.batch_size)/2):
            # randomly choose an identity
            k = self.k
            idx = random.randint(0,len(k)-1)
            # for later use
            list_id_im.append(k[idx])
            list_id_vd.append(k[idx])
            # read the image feature
            img_fea[i] = self.dic_im_face[k[idx]][:,0,0]
            # randomly choose another image from the same identity
            id_dir = self.dataset_root+'/'+ k[idx]
            im_list = os.listdir(id_dir)
            im_list.sort()
            if len(im_list)>1:
                img_name = im_list[random.randint(1,len(im_list)-1)]
            else:
                img_name = im_list[0]
            # fetch the video image
            fetch_name = id_dir+'/'+img_name
            video_img_original[i,:,:,:] = self.fetch_image(fetch_name,0)
            video_img[i,:,:,:] = self.fetch_image(fetch_name,1)
            
            # same flag
            same[i] = 1
        
        # fetch the different pairs
        for i in range((self.batch_size)/2,self.batch_size):
            # randomly choose an identity
            k = self.k
            idx = random.randint(0,len(k)-1)
            idx_diff = self.randint_except(0,idx,len(k)-1)
            # for later use
            list_id_im.append(k[idx])
            list_id_vd.append(k[idx_diff])
            # read the image feature
            img_fea[i] = self.dic_im_face[k[idx]][:,0,0]
            # randomly choose another image from the another identity
            id_dir = self.dataset_root+'/'+ k[idx_diff]
            im_list = os.listdir(id_dir)
            img_name = im_list[random.randint(0,len(im_list)-1)]
            # fetch the video image
            video_img_original[i,:,:,:] = self.fetch_image(id_dir+'/'+img_name,0)
            video_img[i,:,:,:] = self.fetch_image(id_dir+'/'+img_name,1)
            # same flag
            same[i] = 0
            
            
            
        # shuffle
        shuffle_list = []
        for i in range(self.batch_size):
            shuffle_list.append([video_img_original[i,:,:,:],\
                                video_img[i,:,:,:],\
                                img_fea[i,:],\
                                same[i]])
        np.random.shuffle(shuffle_list)
        
        for i in range(self.batch_size):
            video_img_original[i,:,:,:] = shuffle_list[i][0]
            video_img[i,:,:,:] = shuffle_list[i][1]
            img_fea[i,:] = shuffle_list[i][2]
            same[i] = shuffle_list[i][3]
        
        # put data into the top blob
        top[0].data[...] = video_img_original
        top[1].data[...] = video_img
        #top[2].data[...] = img_fea
        #top[3].data[...] = same
        
    def backward(self,top,propagate_down,bottom):
        pass
        
    def fetch_image(self,img_path,resize):
        img = Image.open(img_path)
        if resize == 1:
            img = img.resize((self.input_size,self.input_size))
        in_ = np.array(img,dtype=np.float32)
        #in_ = in_[:,:,::-1]
        in_ -= self.mean
        in_ = in_.transpose((2,0,1))
        return in_
        
    
    def randint_except(self,start,e,end):
        """
        Randomly select an integer from a range(start,end), except one integer e
        """
        assert(start<=end-1)
        tmp = random.randint(start,end-1)
        if(tmp>=e):
            return tmp+1
        else:
            return tmp
        