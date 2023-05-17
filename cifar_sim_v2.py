#import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import models.simsiam as simsiam
import os 
import configparser

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
    

# def compute_features(model, data):
#     feats = model.predict(data)
#     return feats


if __name__ == '__main__' :
    
    ssearch = SSearch('example.ini')
    ssearch.load_model()