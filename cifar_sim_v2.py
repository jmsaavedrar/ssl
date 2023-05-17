#import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import models.simsiam as simsiam
import os 
import configparser
import skimage.io as io

class SSearch():
    def __init__(self, configfile):
        config = configparser.ConfigParser()
        config.read(configfile)
        self.config_model = config['SIMSIAM']
        self.config_data = config['DATA']    
        simsiam_model = simsiam.SimSiam(self.config_data, self.config_model)
        saved_to = os.path.join("saved_model","saved-model")
        simsiam_model.load_weights(saved_to)
        self.model= simsiam_model.encoder
                

    def load_model(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        idx = np.random.permutation(x_test.shape[0])
        self.data = x_test[idx[:1000], :, :, :]        
        print(self.data.shape)
    
    def compute_features(self):
        feats = self.model.predict(self.data)        
        norm = np.linalg.norm(feats, ord = 2, axis = 1, keepdims = True)
        feats = feats / norm
        sim = np.matmul(feats, np.transpose(feats))
        self.sorted_pos = np.argsort(-sim, axis = 1) 


    def visualize(self, sort_idx):    
        size = self.data[1]
        n = 10
        image = np.ones((size, n*size), dtype = np.uint8)*255                        
        i = 0
        for i in np.arange(n) :
            image[:, i * size:(i + 1) * size] = self.data[self.sorted_idx[i], : , : ]
            i = i + 1   
        return image       
         

if __name__ == '__main__' :
    
    ssearch = SSearch('example.ini')
    ssearch.load_model()
    ssearch.compute_features()
    idxs = np.random.randint(1000, 10)
    for idx in idxs :
        rimage =  ssearch.visualize(idx)
        fname = 'result_2_{}.png'.format(idx)
        fname = os.path.join('results',fname)
        io.imsave(fname, rimage)
        print('result saved at {}'.format(fname))
    
     