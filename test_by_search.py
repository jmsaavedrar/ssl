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
    def __init__(self, configfile, model, size = 1000):
        config = configparser.ConfigParser()
        config.read(configfile)
        self.config_model = config[model]
        self.config_data = config['DATA']
        self.model = None 
        self.size = size       
        if  model  == 'SIMSIAM' :
            ssl_model = simsiam.SketchSimSiam(self.config_data, self.config_model)                                            
            ssl_model.load_weights(self.config_model.get('CKP_FILE'))            
            self.model= ssl_model.encoder
        if  model  == 'BYOL' :
            ssl_model = byol.SketchBYOL(self.config_data, self.config_model)
            ssl_model.build()
            ssl_model.load_weights(self.config_model.get('CKP_FILE'))
            self.model= ssl_model.online_encoder
        assert not (self.model == None), '-- there is not a ssl model'
        print('Model loaded OK', flush = True)            

    def load_data(self):
        ds = None
        fn = None
        if self.config_data.get('DATASET') == 'QD' :       
            ds = tfds.load('tfds_qd')
            fn = ssl_map_func
        if self.config_data.get('DATASET') == 'IMAGENET' :       
            ds = tfds.load('imagenet1k')
            fn = imagenet_map_func    
            
        ds_test = ds['test']
        ds_test = ds_test.map(lambda image : fn(image, self.config_data.getint('CROP_SIZE') ))
        ds_test = ds_test.shuffle(1024).batch(self.size)
        ds_test = ds_test.take(1)       
        for sample in ds_test :
            self.data = sample[0].numpy()
            self.labels = sample[1].numpy()
            print(self.data)
            print(self.labels)
              
    
    def compute_map(self):
        labels_ranking = self.labels[self.sorted_pos]
        labels = np.reshape(labels_ranking[:, 0], (-1, 1))
        pos_all_queries = np.where(labels == labels_ranking[:, 1:])
        n_queries = labels_ranking.shape[0]
        AP = 0
        for pos_query in pos_all_queries :
            recall = np.arange(1, labels_ranking.shape[1])
            pr = recall / pos_query
            AP = AP + np.mean(pr)
        mAP = AP / n_queries
        return mAP
        
        return mAP
    def compute_features(self):
        feats = self.model.predict(self.data)        
        norm = np.linalg.norm(feats, ord = 2, axis = 1, keepdims = True)
        feats = feats / norm
        sim = np.matmul(feats, np.transpose(feats))
        self.sorted_pos = np.argsort(-sim, axis = 1)
         


    def visualize(self, idx):
        size = self.config_data.getint('CROP_SIZE')
        n = 10
        image = np.ones((size, n*size, 3), dtype = np.uint8)*255
        i = 0
        for i , pos in enumerate(self.sorted_pos[idx, :n]) :
            image[:, i * size:(i + 1) * size, :] = self.data[pos,:,:,:]
        return image
       

    def get_dataset_name(self):
        return self.config_data.get('DATASET')     

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type = str, required = True)    
    parser.add_argument('-model', type = str, required = True)
    parser.add_argument('-gpu', type = int, required = False) # gpu = -1 set for using all gpus
    datasize = 1000
    args = parser.parse_args()
    gpu_id = 0
    if not args.gpu is None :
        gpu_id = args.gpu
    config_file = args.config
    ssl_model_name = args.model
    assert os.path.exists(config_file), '{} does not exist'.format(config_file)        
    #assert ssl_model_name in ['BYOL', 'SIMSIAM'], '{} is not a valid model'.format(ssl_model_name)
    if gpu_id >= 0 :
        with tf.device('/device:GPU:{}'.format(gpu_id)) :
            ssearch = SSearch(config_file, ssl_model_name, size = datasize)
            ssearch.load_data()
            ssearch.compute_features()
            mAP  = ssearch.compute_map()
            print('mAP = {}'.format(mAP))       
            idxs = np.random.randint(datasize, size = 10)
            dataset_name = ssearch.get_dataset_name()
            result_dir = os.path.join('results', dataset_name, ssl_model_name)
            if not os.path.exists(result_dir) :
                os.makedirs(result_dir)
                 
            for idx in idxs :
                rimage =  ssearch.visualize(idx)
                fname = 'result_{}_{}_{}.png'.format(dataset_name, ssl_model_name, idx)
                fname = os.path.join(result_dir,fname)
                io.imsave(fname, rimage)
                print('result saved at {}'.format(fname))
     