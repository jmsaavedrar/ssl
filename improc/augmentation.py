import tensorflow as tf
import tensorflow_addons as tfa

class DataAugmentation():
    def __init__(self, config : dict):
        self.config = config
                        
    def flip_random_crop(self, image):
        # With random crops we also apply horizontal flipping.
        image = tf.image.random_flip_left_right(image)
        size_resize = int(self.config.getint('CROP_SIZE')*1.15)
        image = tf.image.resize_with_pad(image, size_resize, size_resize)
        image = tf.image.random_crop(image, (self.config.getint('CROP_SIZE'), self.config.getint('CROP_SIZE'), 3))
        return image
        
    def color_jitter(self, x, strength=[0.4, 0.4, 0.4, 0.1]):
        x = tf.image.random_brightness(x, max_delta=0.8 * strength[0])
        x = tf.image.random_contrast(
            x, lower=1 - 0.8 * strength[1], upper=1 + 0.8 * strength[1]
        )
        x = tf.image.random_saturation(
            x, lower=1 - 0.8 * strength[2], upper=1 + 0.8 * strength[2]
        )
        x = tf.image.random_hue(x, max_delta=0.2 * strength[3])
        # Affine transformations can disturb the natural range of
        # RGB images, hence this is needed.
        x = tf.clip_by_value(x, 0, 255)
        return x
    
    
    def color_drop(self, x):
        x = tf.image.rgb_to_grayscale(x)
        x = tf.tile(x, [1, 1, 3])
        return x
    
    
    def random_apply(self, func, x, p):
        if tf.random.uniform([], minval=0, maxval=1) < p:
            return func(x)
        else:
            return x
    
    def custom_augment(self, image):
        # As discussed in the SimCLR paper, the series of augmentation
        # transformations (except for random crops) need to be applied
        # randomly to impose translational invariance.        
        if self.config.get('DATASET') == 'MNIST' :            
            image = tf.image.grayscale_to_rgb(image)
            
        image = self.flip_random_crop(image)
        image = self.random_apply(self.color_jitter, image, p=0.8)
        image = self.random_apply(self.color_drop, image, p=0.2)        
        return image
    
    def sketch_augment(self, image):
        # As discussed in the SimCLR paper, the series of augmentation
        # transformations (except for random crops) need to be applied
        # randomly to impose translational invariance.        
                    
        image = tf.image.grayscale_to_rgb(image)
            
        image = self.flip_random_crop(image)
        image = self.random_apply(self.color_jitter, image, p=0.8)
        image = self.random_apply(self.color_drop, image, p=0.2)        
        return image
    
    def image_augment(self, image):
        # As discussed in the SimCLR paper, the series of augmentation
        # transformations (except for random crops) need to be applied
        # randomly to impose translational invariance.                
        
        image = self.flip_random_crop(image)        
        image = self.random_apply(self.color_jitter, image, p=0.8)
        image = self.random_apply(self.color_drop, image, p=0.2)
        image = tf.expand_dims(image, axis = 0)
        image = tfa.image.random_cutout(image, (14,14), constant_values= 0)
        image = tf.squeeze(image, axis = 0)
        return image
    
    def get_augmentation_fun(self):
        if self.config.get('DATASET') == 'QD' :
            return self.sketch_augment
        
        if self.config.get('DATASET') == 'IMAGENET' :
            return self.image_augment
        
        return None 
    