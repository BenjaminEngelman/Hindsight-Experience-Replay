import numpy as np
import tensorflow as tf
import keras.backend as K

from keras.initializers import RandomUniform
from keras.models import Model
from keras.layers import Input, Dense, Reshape, LSTM, Lambda, BatchNormalization, GaussianNoise, Flatten

def custom_loss(layer, max):

    # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
    def loss(y_true,y_pred):
        loss = K.mean(K.square(y_pred - y_true) , axis=-1)
        loss += K.mean(K.square(y_true / max), axis=1)

        return loss

    # Return a function
    return loss

class Actor:
    """ Actor Network for the DDPG Algorithm
    """

    def __init__(self, inp_dim, out_dim, act_range, lr, tau):
        self.env_dim = inp_dim
        self.act_dim = out_dim
        self.act_range = act_range
        self.tau = tau
        self.lr = lr
        self.model = self.network()
        self.target_model = self.network()
        self.adam_optimizer = self.optimizer()

    def network(self):
        """ Actor Network for Policy function Approximation, using a tanh
        activation for continuous control. We add parameter noise to encourage
        exploration, and balance it with Layer Normalization.
        """
        inp = Input(shape=[self.env_dim,])
        #
        x = Dense(256, activation='relu')(inp)
        x = GaussianNoise(1.0)(x)
        #
        # x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        x = GaussianNoise(1.0)(x)
        #
        out = Dense(self.act_dim, activation='tanh', kernel_initializer=RandomUniform())(x)
        out = Lambda(lambda i: i * self.act_range)(out)
        #
        return Model(inp, out)

    def predict(self, state):
        """ Action prediction
        """
        state = state.reshape(self.env_dim, 1).T
        return self.model.predict(state)

    def target_predict(self, inp):
        """ Action prediction (target network)
        """
        return self.target_model.predict(inp)

    def transfer_weights(self):
        """ Transfer model weights to target model with a factor of Tau
        """
        W, target_W = self.model.get_weights(), self.target_model.get_weights()
        for i in range(len(W)):
            target_W[i] = self.tau * W[i] + (1 - self.tau)* target_W[i]
        self.target_model.set_weights(target_W)

    def train(self, states, actions, grads):
        """ Actor Training
        """
        print(grads)
        self.adam_optimizer([states, grads], )

    def optimizer(self):
        """ Actor Optimizer
        """
        action_gdts = K.placeholder(shape=(None, self.act_dim))
        params_grad = tf.gradients(self.model.output, self.model.trainable_weights, -action_gdts)
        grads = zip(params_grad, self.model.trainable_weights)
        return K.function([self.model.input, action_gdts], [tf.train.AdamOptimizer(self.lr).apply_gradients(grads)][1:])

   

    def save(self, path):
        self.model.save_weights(path + '_actor.h5')

    def load_weights(self, path):
        self.model.load_weights(path)