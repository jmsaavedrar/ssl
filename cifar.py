""" 2023
    SimSiam adapted from https://keras.io/examples/vision/simsiam/
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import improc.augmentation as aug
import configparser
import models.simsiam as simsiam  



AUTO = tf.data.AUTOTUNE
#load configuracion file
config = configparser.ConfigParser()
config.read('example.ini')
config_model = config['SIMSIAM']
config_data = config['DATA']
daug = aug.DataAugmentation(config_data)
 
#loading dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
ssl_ds_one = tf.data.Dataset.from_tensor_slices(x_train)
ssl_ds_one = (
    ssl_ds_one.shuffle(1024, seed=config_model.getint('SEED'))
    .map(daug.custom_augment, num_parallel_calls=AUTO)
    .batch(config_model.getint('BATCH_SIZE'))
    .prefetch(AUTO)
)

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
# sample_images_one = next(iter(ssl_ds_one))
# plt.figure(figsize=(10, 10))
# for n in range(25):
#     ax = plt.subplot(5, 5, n + 1)
#     plt.imshow(sample_images_one[n].numpy().astype("int"))
#     plt.axis("off")
# plt.show()
#
# # Ensure that the different versions of the dataset actually contain
# # identical images.
# sample_images_two = next(iter(ssl_ds_two))
# plt.figure(figsize=(10, 10))
# for n in range(25):
#     ax = plt.subplot(5, 5, n + 1)
#     plt.imshow(sample_images_two[n].numpy().astype("int"))
#     plt.axis("off")
# plt.show()

#training
#resnet = resnet.ResNet([2,2,2,2], [64,128,256,512], 10)

# Create a cosine decay learning scheduler.
num_training_samples = len(x_train)
steps = config_model.getint('EPOCHS') * (num_training_samples // config_model.getint('BATCH_SIZE'))
lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.03, decay_steps=steps
)

# Create an early stopping callback.
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="loss", patience=5, restore_best_weights=True
)

# Compile model and start training.
simsiam = simsiam.SimSiam(config_data, config_model)
simsiam.compile(optimizer=tf.keras.optimizers.SGD(lr_decayed_fn, momentum=0.6))
history = simsiam.fit(ssl_ds, epochs=config_model.getint('EPOCHS'), callbacks=[early_stopping])

# Visualize the training progress of the model.
plt.plot(history.history["loss"])
plt.grid()
plt.title("Negative Cosine Similairty")
plt.show()