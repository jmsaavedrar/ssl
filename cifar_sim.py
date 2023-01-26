import matplotlib.pyplot as plt
import numpy as np
import models.simsiam as simsiam
import os 
import configparser

def load_data():
    def unpickle(file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='latin1')
        return dict

    filename = '/home/vision/smb-datasets/cifar-10-python/cifar-10-batches-py/data_batch_1'
    data = unpickle(filename)
    data = data['data']    
    data = np.reshape(data, (-1,3,32,32));
    data = np.transpose(data, (0,2,3,1))
    return data
    #plt.imshow(data[0,:,:,:])
    #plt.show()


def load_model(configfile):
    config = configparser.ConfigParser()
    config.read(configfile)
    config_model = config['SIMSIAM']
    config_data = config['DATA']    
    simsiam_model = simsiam.SimSiam(config_data, config_model)
    saved_to = os.path.join("saved_model","saved-model")
    simsiam_model.load_weights(saved_to)
    model= simsiam_model.encoder
    return model

def compute_features(model, data):
    feats = model.predict(data)
    return feats


def simsearch(query_feat, data_feat):
    query_feat = query_feat / np.linalg.norm(query_feat, ord = 2, keepdims = True)
    data_feat = data_feat / np.linalg.norm(data_feat, ord = 2, keepdims = True)
    l2 = 2- np.sqrt(np.matmul(data_feat, np.transpose(query_feat)))
    cosine = np.matmul(data_feat, np.transpose(query_feat))
    print(cosine)
    sorted_idx = np.argsort(-cosine, axis = 0)
    return sorted_idx
    
def plot_result(idx, results, queries, data ):
    w = 32
    h = 32
    n = 5
    image = np.zeros((h, w*(n+1),3), dtype = np.uint8)
    image[0:h, 0:w, :] = queries[idx, :, : ,:]
    for i in np.arange(1,n+1) :
        image[0:h, w*i:w*(i+1), :] = data[results[i], :, : ,:]
    plt.imshow(image)
    plt.show()
    
if __name__ == '__main__' :
    configfile = 'example.ini' 
    data = load_data()
    model = load_model(configfile)
    print('computing features')
    feats = compute_features(model, data)
    print(feats)
    print(feats.shape)
    print('simsearch')    
    queries = data[np.random.choice(data.shape[0], 10), :, :, :]
    query_feats =  compute_features(model, queries)
    print(query_feats)
    idx_result = simsearch(query_feats, feats)
    idx = 9    
    plot_result(idx, idx_result[:,idx], queries, data)
        