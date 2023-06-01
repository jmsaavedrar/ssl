""" 2023
    SimSiam adapted from https://keras.io/examples/vision/simsiam/
"""
import sys
sys.path.insert(0,'/home/jsaavedr/Research/git/datasets')
import os
import tensorflow as tf
import tensorflow_datasets as tfds
import improc.augmentation as aug
import configparser
import models.simsiam as simsiam  
import tfds_mnist.tfds_mnist
import numpy as np


def visualize_data(ds_1, ds_2) :
    fig, ax = plt.subplots(5, 6)    
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


def mnist_map_func(image, daug_func):
    image = image['image']    
    return daug_func(image)
        
#---------------------------------------------------------------------------------------
VISUALIZE = False # if false, it runs training
AUTO = tf.data.AUTOTUNE
#load configuracion file
config = configparser.ConfigParser()
config.read('config/mnist.ini')
config_model = config['SIMSIAM']
config_data = config['DATA']
daug = aug.DataAugmentation(config_data)
 
#loading dataset example cifar
ds = tfds.load('tfds_mnist')
ds_train = ds['train']
ssl_ds_one = ds_train
ssl_ds_two = ds_train 

#data one
#SEED is used to keep the same randomization in both ds_one and ds_two
ssl_ds_one = (
    ssl_ds_one.shuffle(1024, seed=config_model.getint('SEED'))
    .map(lambda x: mnist_map_func(x, daug.custom_augment), num_parallel_calls=AUTO)
    .batch(config_model.getint('BATCH_SIZE'))
    .prefetch(AUTO) )

#data two
ssl_ds_two = (
    ssl_ds_two.shuffle(1024, seed=config_model.getint('SEED'))
    .map(lambda x: mnist_map_func(x, daug.custom_augment), num_parallel_calls=AUTO)
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
    simsiam_model = simsiam.SimSiam(config_data, config_model)
    simsiam_model.compile(optimizer=tf.keras.optimizers.SGD(lr_decayed_fn, momentum=0.9))
    history = simsiam_model.fit(ssl_ds, 
                          epochs=config_model.getint('EPOCHS'), 
                          callbacks=[early_stopping])
      
    #predicting
      
      
    #hisitory = simsiam.evaluate(ssl_ds)
    # Visualize the training progress of the model.
    # Extract the backbone ResNet20.
      
    #saving model
    # print('saving model')
    simsiam_model.save_weights(config_model.get('MODEL_NAME'))
    print("model saved to {}".format(config_model.get('MODEL_NAME')))        
    #