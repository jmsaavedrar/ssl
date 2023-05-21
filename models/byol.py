#import models.resnet as resnet
import models.simple as simple
import tensorflow as tf
import configparser
import os
import numpy as np

class BYOL(tf.keras.Model):
    def __init__(self, config_data, config_model):
        super().__init__()        
        self.CROP_SIZE = config_data.getint('CROP_SIZE')
        self.PROJECT_DIM =  config_model.getint('PROJECT_DIM')
        self.WEIGHT_DECAY = config_model.getfloat('WEIGHT_DECAY')
        self.LATENT_DIM  = config_model.getint('LATENT_DIM')        
        self.CHANNELS = 3
        self.STEPS = config_model.getint('STEPS')
        print('{} {} {} {}'.format(self.CROP_SIZE, self.PROJECT_DIM, self.WEIGHT_DECAY, self.LATENT_DIM))
        #defining the BYOL's components
        self.online_encoder = self.get_encoder()
        self.online_predictor = self.get_predictor()
        self.target_encoder = self.get_encoder()
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.step = 0
        
        
                
    def get_encoder(self):
        # Input and backbone.
        inputs = tf.keras.layers.Input((self.CROP_SIZE, self.CROP_SIZE, self.CHANNELS))                
        x = inputs / 127.5 - 1
        #bkbone = resnet.ResNetBackbone([2,2], [64,128])
        bkbone = simple.Backbone()
        x = bkbone(x)   
        # Projection head.
        x = tf.keras.layers.Dense(
            self.PROJECT_DIM, 
            use_bias=False, 
            kernel_regularizer=tf.keras.regularizers.l2()
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Flatten()(x)        
        x = tf.keras.layers.Dense(
            self.PROJECT_DIM, 
            use_bias=False, 
            kernel_regularizer=tf.keras.regularizers.l2(self.WEIGHT_DECAY)
        )(x)
        outputs = tf.keras.layers.BatchNormalization()(x)
        
        return tf.keras.Model(inputs, outputs, name="encoder")


    def get_predictor(self):
        model = tf.keras.Sequential(
            [
                # Note the AutoEncoder-like structure.
                tf.keras.layers.Input((self.PROJECT_DIM,)),
                tf.keras.layers.Dense(
                    self.LATENT_DIM,
                    use_bias=False,
                    kernel_regularizer=tf.keras.regularizers.l2(self.WEIGHT_DECAY),
                ),
                tf.keras.layers.ReLU(),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(self.PROJECT_DIM ),
            ],
            name="predictor",
        )
        return model

    def compute_loss(self, p, z):
        # The authors of SimSiam emphasize the impact of
        # the `stop_gradient` operator in the paper as it
        # has an important role in the overall optimization.
        z = tf.stop_gradient(z)
        p = tf.math.l2_normalize(p, axis=1)
        z = tf.math.l2_normalize(z, axis=1)
        # Negative cosine similarity (minimizing this is
        # equivalent to maximizing the similarity).
        return -tf.reduce_mean(tf.reduce_sum((p * z), axis=1))

    @property
    def metrics(self):
        return [self.loss_tracker]
    
    def set_distrution_strategy(self, strategy):
        self.strategy = strategy
        
    def train_step_byol(self, batch):
            ds_one, ds_two = batch
            z1_target = self.target_encoder(ds_one)
            z2_target = self.target_encoder(ds_two)
            # Forward pass through the encoder and predictor.
            with tf.GradientTape() as tape:
                z1_online = self.online_encoder(ds_one)            
                
                z2_online = self.online_encoder(ds_two)            
                
                p1_online = self.online_predictor(z1_online)
                p2_online = self.online_predictor(z2_online)
                
                                        
                # Note that here we are enforcing the network to match
                # the representations of two differently augmented batches
                # of data.
                loss = self.compute_loss(p1_online, z2_target) / 2 + self.compute_loss(p2_online, z1_target) / 2
    
            # Compute gradients and update the parameters.
            learnable_params = (
                self.online_encoder.trainable_variables + self.online_predictor.trainable_variables
            )
            gradients = tape.gradient(loss, learnable_params)
            self.optimizer.apply_gradients(zip(gradients, learnable_params))
            
            #del tape
            #update weights
            target_encoder_w = self.target_encoder.get_weights()
            online_encoder_w = self.online_encoder.get_weights()
            tau = (np.cos(np.pi* ((self.step + 1)/self.STEPS)) + 1) / 2
            for i in range(len(online_encoder_w)):
                target_encoder_w[i] = tau * target_encoder_w[i] + (1-tau) * online_encoder_w[i]  
            self.target_encoder.set_weights(target_encoder_w)        
            # Monitor loss.
            #self.loss_tracker.update_state(loss)
            self.step += 1               
            return loss
         
    @tf.function
    def dist_train_step(self, dist_batch):      
        per_replica_losses = self.strategy.run(self.train_step_byol, args=(dist_batch,))
        return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                               axis=None)    
      
    def fit_byol(self, data, epochs):
        dist_dataset = self.strategy.experimental_distribute_dataset(data)
        assert tf.distribute.get_replica_context() is not None
        for epoch in range(epochs) :
            for dist_batch in dist_dataset :                
                    loss = self.dist_train_step(dist_batch)                
                    print('step : {} loss {}'.format(self.step,loss))
            print('epoch : {}'.format(epoch))
#                self.step = self.step + 1 
                #return {"loss": self.loss_tracker.result()}
        
    
if __name__ == '__main__' :    
    config = configparser.ConfigParser()
    config.read('example.ini')
    config_model = config['SIMSIAM']
    config_data = config['DATA']
    simsiam = BYOL(config_data, config_model)        
    simsiam.load_weights(config_data.get('MODEL_NAME'))
    for v in simsiam.encoder.trainable_variables :
        print(v.numpy)