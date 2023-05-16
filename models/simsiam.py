import models.resnet as resnet
import models.simple as simple
import tensorflow as tf


class SimSiam(tf.keras.Model):
    def __init__(self, config_data, config_model):
        super().__init__()        
        self.CROP_SIZE = config_data.getint('CROP_SIZE')
        self.PROJECT_DIM =  config_model.getint('PROJECT_DIM')
        self.WEIGHT_DECAY = config_model.getfloat('WEIGHT_DECAY')
        self.LATENT_DIM  = config_model.getint('LATENT_DIM')
        print('{} {} {} {}'.format(self.CROP_SIZE, self.PROJECT_DIM, self.WEIGHT_DECAY, self.LATENT_DIM))
        self.encoder = self.get_encoder()
        self.predictor = self.get_predictor()
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        
                
    def get_encoder(self):
        # Input and backbone.
        inputs = tf.keras.layers.Input((self.CROP_SIZE, self.CROP_SIZE, 3))                
        x = inputs / 127.5 - 1
        #bkbone = resnet.ResNetBackbone([2,2], [64,128])
        bkbone = simple.BackBone()
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
                tf.keras.layers.Dense(self.PROJECT_DIM),
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

    def train_step(self, data):
        # Unpack the data.
        ds_one, ds_two = data

        # Forward pass through the encoder and predictor.
        with tf.GradientTape() as tape:
            z1, z2 = self.encoder(ds_one), self.encoder(ds_two)
            p1, p2 = self.predictor(z1), self.predictor(z2)
            # Note that here we are enforcing the network to match
            # the representations of two differently augmented batches
            # of data.
            loss = self.compute_loss(p1, z2) / 2 + self.compute_loss(p2, z1) / 2

        # Compute gradients and update the parameters.
        learnable_params = (
            self.encoder.trainable_variables + self.predictor.trainable_variables
        )
        gradients = tape.gradient(loss, learnable_params)
        self.optimizer.apply_gradients(zip(gradients, learnable_params))

        # Monitor loss.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}
