import tensorflow as tf
import numpy as np
import math
import cv2
from scipy import misc
import MobileFaceNet as mobilenet

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y  

def crop(image, random_crop, image_size):
    if image.shape[1]>image_size:
        sz1 = int(image.shape[1]//2)
        sz2 = int(image_size//2)
        if random_crop:
            diff = sz1-sz2
            (h, v) = (np.random.randint(-diff, diff+1), np.random.randint(-diff, diff+1))
        else:
            (h, v) = (0,0)
        image = image[(sz1-sz2+v):(sz1+sz2+v),(sz1-sz2+h):(sz1+sz2+h),:]
    return image
  
def flip(image, random_flip):
    if random_flip and np.random.choice([True, False]):
        image = np.fliplr(image)
    return image

def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret
  



def load_data(image_paths, do_random_crop, do_flip, image_height,image_width, do_prewhiten=True,src_size=None):
    nrof_samples = len(image_paths)
    images = np.zeros((nrof_samples, image_height, image_width, 3))
    for i in range(nrof_samples):
        #img = misc.imread(image_paths[i])
        img = cv2.imread(image_paths[i])[:,:,::-1]
        if src_size is not None:
            #img = misc.imresize(img,(src_size[0],src_size[1]))
            img = cv2.resize(img,(src_size[0],src_size[1]))
        if img.ndim == 2:
            img = to_rgb(img)
        if do_prewhiten:
            img = prewhiten(img)
        else:
            img = img - 127.5
            img = img / 128.

        img = crop(img, do_random_crop, image_width)
        #img = flip(img, do_random_flip)
        if do_flip:
            img = np.fliplr(img)
        images[i,:,:,:] = img
    return images

def l2_normalize(x):
    n,e = x.shape
    mean = np.mean(x,axis=1)
    mean = mean.reshape((n,1))
    mean = np.repeat(mean,e,axis=1)
    x -= mean
    norm = np.linalg.norm(x,axis=1)
    norm = norm.reshape((n,1))
    norm = np.repeat(norm,e,axis=1)
    y = np.multiply(x,1/norm)
    return y


class facenet_encoder:
    def __init__(self,ckpt_file,embedding_size=1024):
        self.image_height = 112
        self.image_width = 112
        self.embedding_size = embedding_size
        
        self.images_placeholder = tf.placeholder(tf.float32,shape=(None,self.image_height,self.image_width,3),name='image')
        self.phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
        prelogits, net_points = mobilenet.inference(self.images_placeholder,bottleneck_layer_size=self.embedding_size,phase_train=self.phase_train_placeholder,weight_decay=0.0005,reuse=False)
        self.embeddings = prelogits
        self.sess = tf.Session()
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=3)
        saver.restore(self.sess,ckpt_file)
    def generate_batch_embeddings(self,paths,batch_size=200):
        nrof_images = len(paths)
        emb_array = np.zeros((nrof_images, self.embedding_size))
        nrof_batches = int(math.ceil(1.0*nrof_images / batch_size))
        for i in range(nrof_batches):
            start_index = i*batch_size
            print('handing {}/{}'.format(start_index,nrof_images))
            end_index = min((i+1)*batch_size, nrof_images)
            paths_batch = paths[start_index:end_index]
            images = load_data(paths_batch, False, False, self.image_height,self.image_width,False,(self.image_height,self.image_width))
            feed_dict = { self.images_placeholder:images, self.phase_train_placeholder:False }
            feats = self.sess.run(self.embeddings, feed_dict=feed_dict)

            feats = l2_normalize(feats)
            emb_array[start_index:end_index,:] = feats
        return emb_array
    def generate_single_embedding(self,image_path):
        emb_array = self.generate_batch_embeddings([image_path],batch_size=1) 
        return emb_array
    
