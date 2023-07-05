"""
This scripts train a ssl model. So far we have

-- SimSiam <adapted from https://keras.io/examples/vision/simsiam/>
-- BYOL

This code use tfds to prepare and load the datasets. Our examepl is with
QuickDraw datasets, so you need the following repository

"""


import sys
import socket
ip = socket.gethostbyname(socket.gethostname())
## you can modify this part to set a local path for the dataset project  
if ip == '192.168.20.62' :
    sys.path.insert(0,'/home/DIINF/vchang/jsaavedr/Research/git/datasets')
else :
    sys.path.insert(0,'/home/jsaavedr/Research/git/datasets')
import os
import tensorflow as tf
import tensorflow_datasets as tfds
import improc.augmentation as aug
import configparser
import models.sketch_simsiam as simsiam  
import models.sketch_byol as byol
import argparse
# import the dataset builder, here is an example for qd


def visualize_data(ds_1, ds_2) :
    _, ax = plt.subplots(5, 6)    
    for _ in range(5):
        sample_images_one = next(iter(ds_1))
        sample_images_two = next(iter(ds_2))
        print(sample_images_one.shape)
        for i in range(15):        
            ax[i % 5][(i // 5)*2].imshow(sample_images_one[i].numpy().astype("int"))
            
            ax[i % 5][((i) // 5)*2 + 1].imshow(sample_images_two[i].numpy().astype("int"))
            plt.axis("off")
        plt.waitforbuttonpress(1)
    plt.show()

#---------------------------------------------------------------------------------------
def ssl_map_func(image, daug_func):
    image = image['image']    
    return daug_func(image)
        
#---------------------------------------------------------------------------------------
VISUALIZE = False # if false, it runs training
if ip == '127.0.1.1' :
    VISUALIZE = True

AUTO = tf.data.AUTOTUNE
#---------------------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type = str, required = True)    
    parser.add_argument('-model', type = str, required = True)
    parser.add_argument('-gpu', type = int, required = False)
    args = parser.parse_args()
    gpu_id = 0
    if not args.gpu is None :
        gpu_id = args.gpu    
    config_file = args.config
    ssl_model_name = args.model
    assert os.path.exists(config_file), '{} does not exist'.format(config_file)        
    assert ssl_model_name in ['BYOL', 'SIMSIAM'], '{} is not a valid model'.format(ssl_model_name)
    #load configuracion file
    config = configparser.ConfigParser()
    config.read(config_file)
    config_model = config[ssl_model_name]
    assert not config_model == None, '{} does not exist'.format(ssl_model_name)
    
    config_data = config['DATA']
    ds = None
    #
    if config_data.get('DATASET') == 'QD' :        
        ds = tfds.load('tfds_qd')
    if config_data.get('DATASET') == 'IMAGENET' :        
        data_dir ='/mnt/hd-data/Datasets/imagenet/tfds'            
        ds = tfds.load('imagenet1k', data_dir = data_dir)
        
    daug = aug.DataAugmentation(config_data)    
    #loading dataset example cifar
    
    ds_train = ds['train']
    ssl_ds_one = ds_train
    ssl_ds_two = ds_train     
    #data one
    #SEED is used to keep the same randomization in both ds_one and ds_two
    ssl_ds_one = (
        ssl_ds_one.shuffle(1024, seed=config_model.getint('SEED'))
        .map(lambda x: ssl_map_func(x, daug.get_augmentation_fun()), num_parallel_calls=AUTO)
        .batch(config_model.getint('BATCH_SIZE'))
        .prefetch(AUTO) )
    
    #data two
    ssl_ds_two = (
        ssl_ds_two.shuffle(1024, seed=config_model.getint('SEED'))
        .map(lambda x: ssl_map_func(x, daug.get_augmentation_fun()), num_parallel_calls=AUTO)
        .batch(config_model.getint('BATCH_SIZE'))
        .prefetch(AUTO))
    
    # We then zip both of these datasets to have only one
    ssl_ds = tf.data.Dataset.zip((ssl_ds_one, ssl_ds_two))
    
    # # Visualize a few augmented images.
    if VISUALIZE :
        import matplotlib.pyplot as plt
        visualize_data(ssl_ds_one, ssl_ds_two)
    else :
        # Create a cosine decay learning scheduler.
        num_training_samples = len(ds_train)
        
        steps = config_model.getint('EPOCHS') * (num_training_samples // config_model.getint('BATCH_SIZE'))
        config_model['STEPS'] = str(steps)
        lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=0.03, decay_steps = steps)
          
        # Create an early stopping callback.
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="loss", patience=5, restore_best_weights=True
        )
          
        # Compile model and start training.
        with tf.device('/device:GPU:{}'.format(gpu_id)) :
            if ssl_model_name == 'SIMSIAM' :
                ssl_model = simsiam.SketchSimSiam(config_data, config_model)
                ssl_model.compile(optimizer=tf.keras.optimizers.SGD(lr_decayed_fn, momentum=0.9))
                history = ssl_model.fit(ssl_ds, 
                                  epochs=config_model.getint('EPOCHS'), 
                                  callbacks=[early_stopping])
                
            if ssl_model_name == 'BYOL' :
                ssl_model = byol.SketchBYOL(config_data, config_model)
                ssl_model.compile(optimizer=tf.keras.optimizers.SGD(lr_decayed_fn, momentum=0.9))
                history = ssl_model.fit_byol(ssl_ds, epochs=config_model.getint('EPOCHS'))
                              
        #predicting
                    
        #hisitory = simsiam.evaluate(ssl_ds)
        # Visualize the training progress of the model.
        # Extract the backbone ResNet20.
        #saving model
        # print('saving model')
        model_file =config_model.get('MODEL_NAME')
        if not os.path.exists(os.path.dirname(model_file)) :
            os.makedirs(os.path.dirname(model_file))
            print('--- {} was created'.format(os.path.dirname(model_file)))
        ssl_model.save_weights(model_file)
        print("model saved to {}".format(model_file))        
    #