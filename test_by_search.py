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
import os 
import configparser
import skimage.io as io
import tensorflow_datasets as tfds
import argparse
#---------- dataset builder --------------------  
import tfds_qd.tfds_qd as tfds_qd

def mnist_map_func(image, crop_size):
    image = image['image']    
    image = tf.image.grayscale_to_rgb(image) 
    image = tf.image.resize(image, (crop_size,crop_size))
    return image
        
        
class SSearch():
    def __init__(self, configfile, model):
        config = configparser.ConfigParser()
        config.read(configfile)
        self.config_model = config[model]
        self.config_data = config['DATA']
        self.model = None
        
        if  model  == 'SIMSIAM' :
            ssl_model = simsiam.SketchSimSiam(self.config_data, self.config_model)        
            ssl_model.load_weights(self.config_model.get('MODEL_NAME'))
            self.model= ssl_model.encoder
        if  model  == 'BYOL' :
            ssl_model = byol.SketchBYOL(self.config_data, self.config_model)        
            ssl_model.load_weights(self.config_model.get('MODEL_NAME')) 
            self.model= ssl_model.online_encoder
        assert not (self.model == None), '-- there is not a ssl model'        

    def load_data(self):            
        ds = tfds.load('tfds_qd')
        ds_test = ds['test'].shuffle(1024).batch(1000)
        ds_test = ds_test.map(lambda image : mnist_map_func(image, self.config_data.getint('CROP_SIZE') ))
        ds_test = ds_test.take(1)
        for sample in ds_test :
            self.data = sample.numpy()
              
    
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
       
    
    def get_dataset(self):
        return self.config_data.get('DATASET')     

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type = str, required = True)    
    parser.add_argument('-model', type = str, required = True)
    args = parser.parse_args()
    config_file = args.config
    ssl_model_name = args.model
    assert os.path.exists(config_file), '{} does not exist'.format(config_file)        
    assert ssl_model_name in ['BYOL', 'SIMSIAM'], '{} is not a valid model'.format(ssl_model_name)
    ssearch = SSearch(config_file, ssl_model_name)
    ssearch.load_data()
    ssearch.compute_features()
    idxs = np.random.randint(1000, size = 10)
    dataset = ssearch.get_dataset()
    result_dir = os.path.join('results', ssl_model_name)
    if not os.path.exists(result_dir) :
        os.makedirs(result_dir)
        
    for idx in idxs :
        rimage =  ssearch.visualize(idx)
        fname = 'result_{}_{}_{}.png'.format(dataset, ssl_model_name, idx)
        fname = os.path.join(result_dir,fname)
        io.imsave(fname, rimage)
        print('result saved at {}'.format(fname))
    