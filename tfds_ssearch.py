#import matplotlib.pyplot as plt
import sys
sys.path.insert(0,'/home/jsaavedr/Research/git/datasets')
import tfds_mnist.tfds_mnist as tfds_mnist
import tensorflow as tf
import numpy as np
import models.simsiam as simsiam
import os 
import configparser
import skimage.io as io
import tensorflow_datasests as tfds

def mnist_map_func(image, daug_func):
    image = image['image']    
    image = tf.image.grayscale_to_rgb(image) 
    return image
        
        
class SSearch():
    def __init__(self, configfile, model):
        config = configparser.ConfigParser()
        config.read(configfile)
        self.config_model = config[model]
        self.config_data = config['DATA']    
        simsiam_model = simsiam.SimSiam(self.config_data, self.config_model)        
        simsiam_model.load_weights(self.config_model.get('MODEL_NAME'))
        self.model= simsiam_model.encoder
                

    def load_data(self):
        dataset = self.config_data.get('DATASET')
        assert (dataset in ['CIFAR', 'MNIST']), 'dataset is not available'
        ds = tfds.load('tfds_mnist')
        ds_test = ds['test'].shuffle(1024)
        ds_test = ds_test.take(1000)        
        self.data = ds_test                
                                
    
    def compute_features(self):
        feats = self.model.predict(self.data)        
        norm = np.linalg.norm(feats, ord = 2, axis = 1, keepdims = True)
        feats = feats / norm
        sim = np.matmul(feats, np.transpose(feats))
        self.sorted_pos = np.argsort(-sim, axis = 1) 


    def visualize(self, idx):    
        size = self.data.shape[1]        
        n = 10
        image = np.ones((size, n*size, 3), dtype = np.uint8)*255                        
        i = 0
        for i , pos in enumerate(self.sorted_pos[idx, :n]) :
            image[:, i * size:(i + 1) * size, :] = self.data[pos, : , :, : ]
               
        return image       
    
    def get_dataset(self):
        return self.config_data.get('DATASET')     

if __name__ == '__main__' :
    
    ssearch = SSearch('config/mnist.ini', 'SIMSIAM')
    ssearch.load_data()
    ssearch.compute_features()
    idxs = np.random.randint(1000, size = 10)
    dataset = ssearch.get_dataset()
    for idx in idxs :
        rimage =  ssearch.visualize(idx)
        fname = 'result_{}_{}.png'.format(dataset, idx)
        fname = os.path.join('results',fname)
        io.imsave(fname, rimage)
        print('result saved at {}'.format(fname))
    