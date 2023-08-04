"""
Thi scripst tests a ssl model using similarity search
Models :
-- SimSiam
-- BYOL
"""
import sys
import socket
ip = socket.gethostbyname(socket.gethostname())
if ip == '192.168.20.62' :
    sys.path.insert(0,'/home/DIINF/vchang/jsaavedr/Research/git/datasets')
else :
    sys.path.insert(0,'/home/jsaavedr/Research/git/datasets')
import os
import tensorflow as tf
import numpy as np
import models.sketch_simsiam as simsiam
import models.sketch_byol as byol
import configparser
import tensorflow_datasets as tfds
import skimage.io as io
import argparse
#---------- dataset builder --------------------  

#set seed for experimental replication
tf.random.set_seed(1234)


def ssl_map_func(image_label, crop_size):
    image = image_label['image']
    label = image_label['label']    
    image = tf.image.grayscale_to_rgb(image) 
    image = tf.image.resize(image, (crop_size,crop_size))
    return image, label
        
def imagenet_map_func(image_label, crop_size):
    image = image_label['image']
    label = image_label['label']
    #image = tf.image.grayscale_to_rgb(image) 
    size = int(crop_size * 1.15)
    image = tf.image.resize_with_pad(image, size, size)
    image = tf.image.random_crop(image, size = [crop_size, crop_size, 3])
    image = tf.cast(image, tf.uint8) 
    return image, label    

        
class SSearch():
    def __init__(self, configfile, model):
        config = configparser.ConfigParser()
        config.read(configfile)
        self.config_model = config[model]
        self.config_data = config['DATA']
        self.model = None 
        if  model  == 'SIMSIAM' :
            ssl_model = simsiam.SketchSimSiam(self.config_data, self.config_model)                                            
            ssl_model.load_weights(self.config_model.get('CKP_FILE'))            
            self.model= ssl_model.encoder
        if  model  == 'BYOL' :
            ssl_model = byol.SketchBYOL(self.config_data, self.config_model)            
            ssl_model.load_weights(self.config_model.get('CKP_FILE'))
            self.model= ssl_model.online_encoder
        assert not (self.model == None), '-- there is not a ssl model'
        print('Model loaded OK', flush = True)            

    def load_data(self, data_name='test'):
        ds = None
        fn = None
        if self.config_data.get('DATASET') == 'QD' :       
            ds = tfds.load('tfds_qdssl')
            fn = ssl_map_func
        if self.config_data.get('DATASET') == 'IMAGENET' :       
            ds = tfds.load('imagenet1k')
            fn = imagenet_map_func    
            
        ds_test = ds[data_name]
        ds_test = ds_test.map(lambda image : fn(image, self.config_data.getint('CROP_SIZE') ))        
        self.ds_data = ds_test.shuffle(1024).batch(1024)
        
              
    
    def compute_map(self, n_retrieved = -1):                                         
        print(self.labels.shape)                        
        AP = []
        sorted_pos_limited = self.sorted_pos[:, 1:] if n_retrieved == -1 else self.sorted_pos[:, 1:n_retrieved + 1] 
        for i in np.arange(self.get_dataset_size()) :
            ranking = self.labels[sorted_pos_limited[i,:]]                 
            pos_query = np.where(ranking == self.labels[i])[0]
            pos_query = pos_query + 1 
            if len(pos_query) == 0 :
                AP_q = 0
            else :
                recall = np.arange(1, len(pos_query) + 1)
                pr = recall / pos_query
                AP_q = np.mean(pr)
            AP.append(AP_q)
            print('{} -> mAP = {}'.format(len(pos_query), AP_q))
                         
        mAP = np.mean(np.array(AP))        
        return mAP
        
        
    def compute_features(self):
        self.features = np.array([])
        self.labels = np.array([])
        #self.images = np.array([])
        for batch in self.ds_data :          
            images = batch[0].numpy()            
            labels = batch[1].numpy()                
            feats = self.model.predict(images)
          #  imgs = np.array([np.resize(im, (32,32)) for im in images])
            self.features = np.vstack([self.features, feats]) if self.features.size else feats
            self.labels = np.vstack([self.labels, labels]) if self.labels.size else labels
         #   self.images= np.vstack([self.images, imgs]) if self.images.size else imgs
        self.labels = np.reshape(self.labels, (-1,))
        
    
    def compute_sim(self):        
        feats = self.features
        print(feats.shape)
        norm = np.linalg.norm(feats, ord = 2, axis = 1, keepdims = True)
        feats = feats / norm
        sim = np.matmul(feats, np.transpose(feats))
        print(sim.shape)
        sim_sample = sim[np.random.choice(np.arange(sim.shape[0]), size = 1000), :]
        self.sorted_pos = np.argsort(-sim_sample, axis = 1)
        print(self.sorted_pos.shape)
         


    def visualize(self, idx):
        size = self.config_data.getint('CROP_SIZE')
        n = 10
        image = np.ones((size, n*size, 3), dtype = np.uint8)*255
        i = 0
        for i , pos in enumerate(self.sorted_pos[idx, :n]) :
            image[:, i * size:(i + 1) * size, :] = self.images[pos,:,:,:]
        return image
       

    def get_dataset_name(self):
        return self.config_data.get('DATASET')
    
    def get_exp_id(self):
        return self.config_model.get('EXP_CODE')
    
    def get_dataset_size(self):
        return len(self.labels)     

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type = str, required = True)    
    parser.add_argument('-model', type = str, required = True)
    parser.add_argument('-gpu', type = int, required = False) # gpu = -1 set for using all gpus
    parser.add_argument('-save_sample', type = bool, required = False, default = False, action=argparse.BooleanOptionalAction) # gpu = -1 set for using all gpus
    parser.add_argument('-data_name', type = str, required = False, default = 'test') # the name of the testing data ej. test, test_known, test_unknown
    #datasize = 1000
    args = parser.parse_args()
    gpu_id = 0
    data_name = 'test'    
    if not args.gpu is None :
        gpu_id = args.gpu
    if not args.data_name is None :
        data_name = args.data_name
        
    config_file = args.config
    ssl_model_name = args.model
    assert os.path.exists(config_file), '{} does not exist'.format(config_file)        
    #assert ssl_model_name in ['BYOL', 'SIMSIAM'], '{} is not a valid model'.format(ssl_model_name)
    if gpu_id >= 0 :
        with tf.device('/device:GPU:{}'.format(gpu_id)) :
            ssearch = SSearch(config_file, ssl_model_name)
            ssearch.load_data(data_name)
            ssearch.compute_features()
            ssearch.compute_sim()
            mAP  = ssearch.compute_map(n_retrieved = 5)
            print('mAP \t = {}'.format(mAP))
            datasize = ssearch.get_dataset_size()
            print('dataset size \t = {}'.format(datasize))
            if args.save_sample : 
                idxs = np.random.randint(datasize, size = 10)
                dataset_name = ssearch.get_dataset_name()
                exp_id = ssearch.get_exp_id()
                result_dir = os.path.join('results', exp_id, dataset_name, ssl_model_name)
                if not os.path.exists(result_dir) :
                    os.makedirs(result_dir)
                        
                for idx in idxs :
                    rimage =  ssearch.visualize(idx)
                    fname = 'result_{}_{}_{}.png'.format(dataset_name, ssl_model_name, idx)
                    fname = os.path.join(result_dir,fname)
                    io.imsave(fname, rimage)
                    print('result saved at {}'.format(fname))
     