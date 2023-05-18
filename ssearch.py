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
        self.config_model = config['BYOL']
        self.config_data = config['DATA']    
        simsiam_model = simsiam.SimSiam(self.config_data, self.config_model)        
        simsiam_model.load_weights(self.config_model.get('MODEL_NAME'))
        self.model= simsiam_model.encoder
                

    def load_data(self):
        dataset = self.config_data.get('DATASET')
        assert (dataset in ['CIFAR', 'MNIST']), 'dataset is not available'
        if  dataset == 'CIFAR' :
            (_, _), (x_test, _) = tf.keras.datasets.cifar10.load_data()
        if dataset == 'MNIST' : 
            (_, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
            x_test = np.expand_dims(x_test, axis = -1)
            x_test = np.concatenate((x_test, x_test, x_test), axis = 3)
            
        idx = np.random.permutation(x_test.shape[0])
        self.data = x_test[idx[:1000], :, :, :]        
        print(self.data.shape)
        
    def load_data2(self):
        
        def unpickle(file):
            import pickle
            with open(file, 'rb') as fo:
                dict = pickle.load(fo, encoding='latin1')
            return dict
        filename = './cifar/cifar-10-batches-py/data_batch_1'
        """
        cifar 10 url        
        https://www.cs.toronto.edu/~kriz/cifar.html 
        """
        data = unpickle(filename)
        data = data['data']    
        data = np.reshape(data, (-1,3,32,32));
        data = np.transpose(data, (0,2,3,1))
        idx = np.random.permutation(data.shape[0])
        self.data = data[idx[:1000], :, :, :]                
    
    
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
         

if __name__ == '__main__' :
    
    ssearch = SSearch('example.ini')
    ssearch.load_data()
    ssearch.compute_features()
    idxs = np.random.randint(1000, size = 10)
    for idx in idxs :
        rimage =  ssearch.visualize(idx)
        fname = 'result_datos_2_{}.png'.format(idx)
        fname = os.path.join('results',fname)
        io.imsave(fname, rimage)
        print('result saved at {}'.format(fname))
    
     