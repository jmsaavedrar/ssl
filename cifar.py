""" 2023
    SimSiam adapted from https://keras.io/examples/vision/simsiam/
"""
import tensorflow as tf
import improc.augmentation as aug
import configparser
import models.simsiam as simsiam  
import os
import matplotlib.pyplot as plt

def visualize_data(ds_1, ds_2) :
    
    
    #plt.figure(figsize=(10, 10))
    fig, ax = plt.subplots(6, 5)
    
    for _ in range(5):
        sample_images_one = next(iter(ds_1))
        sample_images_two = next(iter(ds_2))
        print(sample_images_one.shape)
        for i in range(15):        
            ax[i//5][i % 5].imshow(sample_images_one[i].numpy().astype("int"))
            ax[i//5 + 3][i % 5].imshow(sample_images_two[i].numpy().astype("int"))
            plt.axis("off")
        plt.waitforbuttonpress(1)
    plt.show()
    
    


AUTO = tf.data.AUTOTUNE
#load configuracion file
config = configparser.ConfigParser()
config.read('example.ini')
config_model = config['SIMSIAM']
config_data = config['DATA']
daug = aug.DataAugmentation(config_data)
 
#loading dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
#data one
ssl_ds_one = tf.data.Dataset.from_tensor_slices(x_train)
ssl_ds_one = (
    ssl_ds_one.shuffle(1024, seed=config_model.getint('SEED'))
    .map(daug.custom_augment, num_parallel_calls=AUTO)
    .batch(config_model.getint('BATCH_SIZE'))
    .prefetch(AUTO)
)

#data two
ssl_ds_two = tf.data.Dataset.from_tensor_slices(x_train)
ssl_ds_two = (
    ssl_ds_two.shuffle(1024, seed=config_model.getint('SEED'))
    .map(daug.custom_augment, num_parallel_calls=AUTO)
    .batch(config_model.getint('BATCH_SIZE'))
    .prefetch(AUTO)
)

# We then zip both of these datasets.
ssl_ds = tf.data.Dataset.zip((ssl_ds_one, ssl_ds_two))

# # Visualize a few augmented images.
#visualize_data(ssl_ds_one, ssl_ds_two)
#training
# #resnet = resnet.ResNet([2,2,2,2], [64,128,256,512], 10)
#  
# Create a cosine decay learning scheduler.
num_training_samples = len(x_train)

steps = config_model.getint('EPOCHS') * (num_training_samples // config_model.getint('BATCH_SIZE'))
lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.03, decay_steps = steps)
  
# Create an early stopping callback.
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="loss", patience=5, restore_best_weights=True
)
  
# Compile model and start training.
simsiam_model = simsiam.SimSiam(config_data, config_model)
simsiam_model.compile(optimizer=tf.keras.optimizers.SGD(lr_decayed_fn, momentum=0.9))
saved_to = os.path.join("saved_model","saved-model")
#simsiam.load_weights(saved_to)
history = simsiam_model.fit(ssl_ds, 
                      epochs=config_model.getint('EPOCHS'), 
                      callbacks=[early_stopping])
  
#predicting
  
  
#hisitory = simsiam.evaluate(ssl_ds)
# Visualize the training progress of the model.
# Extract the backbone ResNet20.
  
#saving model
# print('saving model')
# saved_to = os.path.join("saved_model","saved-model")
simsiam_model.save_weights(saved_to)
print("model saved to {}".format(saved_to))        
#)
#plt.plot(history.history["loss"])
#plt.grid()
#plt.title("Negative Cosine Similairty")
#plt.show()