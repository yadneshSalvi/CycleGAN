import os
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa

#we define a general discriminator model which we will use afterwards in the cyclegan
def discriminator_model(input_image_shape):
  initialization = keras.initializers.RandomNormal(stddev = 0.02)

  input_layer = keras.layers.Input(shape=input_image_shape)

  #there are several conv2D-InstanceNorm-LeakyReLU layers are said in the paper
  l = layers.Conv2D(filters=64, kernel_size=4, strides=2, padding='same', kernel_initializer=initialization)(input_layer)
  l = layers.LeakyReLU(alpha=0.2)(l)

  l = layers.Conv2D(filters=128, kernel_size=4, strides=2, padding='same', kernel_initializer=initialization)(l)
  l = tfa.layers.InstanceNormalization(axis=-1)(l)
  l = layers.LeakyReLU(alpha=0.2)(l)

  l = layers.Conv2D(filters=256, kernel_size=4, strides=2, padding='same', kernel_initializer=initialization)(l)
  l = tfa.layers.InstanceNormalization(axis=-1)(l)
  l = layers.LeakyReLU(alpha=0.2)(l)

  l = layers.Conv2D(filters=512, kernel_size=4, strides=2, padding='same', kernel_initializer=initialization)(l)
  l = tfa.layers.InstanceNormalization(axis=-1)(l)
  l = layers.LeakyReLU(alpha=0.2)(l)

  l = layers.Conv2D(filters=512, kernel_size=4, padding='same', kernel_initializer=initialization)(l)
  l = tfa.layers.InstanceNormalization(axis=-1)(l)
  l = layers.LeakyReLU(alpha=0.2)(l)
  output_layer = layers.Conv2D(filters=1, kernel_size=4, padding='same', kernel_initializer=initialization)(l)

  model = keras.models.Model(input_layer, output_layer)

  model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=0.0002, beta_1=0.5), loss_weights=[0.5])
  return model

#we define a residual block to use it in the generator model
#the output of residual block is f(x) + x
def residual_block(input_layer):
  initialization = keras.initializers.RandomNormal(stddev = 0.02)

  l = layers.Conv2D(256, (3,3), padding='same', kernel_initializer=initialization)(input_layer)
  l = tfa.layers.InstanceNormalization(axis=-1)(l)
  l = layers.Activation('relu')(l)
  
  l = layers.Conv2D(256, (3,3), padding='same', kernel_initializer=initialization)(l)
  l = tfa.layers.InstanceNormalization(axis=-1)(l)
  
  l = layers.Concatenate()([l, input_layer])
  return l

#we define a general generator model which we will use afterwards in the cyclegan
def generator_model(input_image_shape):
  initialization = keras.initializers.RandomNormal(stddev = 0.02)
  
  input_layer = keras.layers.Input(shape=input_image_shape)

  #first we downsample the image using several conv2D layers of stride2
  l = layers.Conv2D(64, (7,7), padding='same', kernel_initializer=initialization)(input_layer)
  l = tfa.layers.InstanceNormalization(axis=-1)(l)
  l = layers.Activation('relu')(l)

  l = layers.Conv2D(128, (3,3), strides=(2,2), padding='same', kernel_initializer=initialization)(l)
  l = tfa.layers.InstanceNormalization(axis=-1)(l)
  l = layers.Activation('relu')(l)

  l = layers.Conv2D(256, (3,3), strides=(2,2), padding='same', kernel_initializer=initialization)(l)
  l = tfa.layers.InstanceNormalization(axis=-1)(l)
  l = layers.Activation('relu')(l)

  #then we apply several residual block layers (ResNet) which will transform the image
  for i in range(9): #add 9 residual blocks
    l = residual_block(l)

  #then we upscale the image usng conv2DTranspose
  l = layers.Conv2DTranspose(128, (3,3), strides=(2,2), padding='same', kernel_initializer=initialization)(l)
  l = tfa.layers.InstanceNormalization(axis=-1)(l)
  l = layers.Activation('relu')(l)

  l = layers.Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', kernel_initializer=initialization)(l)
  l = tfa.layers.InstanceNormalization(axis=-1)(l)
  l = layers.Activation('relu')(l)

  l = layers.Conv2D(3, (7,7), padding='same', kernel_initializer=initialization)(l)
  l = tfa.layers.InstanceNormalization(axis=-1)(l)
  output_layer = layers.Activation('tanh')(l)

  model = keras.models.Model(input_layer, output_layer)
  return model

#finally we define a model for updating our generators
def generator_update_model(generator_horse, discriminator_model, generator_zebra, image_shape):
  generator_horse.trainable = True
  discriminator_model.trainable = False
  generator_zebra.trainable = False

  #first get the output of the discriminator
  generator_input = keras.layers.Input(shape=image_shape)
  generator_horse_output = generator_horse(generator_input)
  discriminator_output = discriminator_model(generator_horse_output)

  #we calculate identity loss as is defined in the official implementation
  identity_input = keras.layers.Input(shape=image_shape)
  identity_output = generator_horse(identity_input)

  #now we get output of the forward and backward cycles
  forward_cycle = generator_zebra(generator_horse_output)#forward cycle
  generator_zebra_out = generator_zebra(identity_input)#backward cycle
  backward_cycle = generator_horse(generator_zebra_out)

  model = keras.models.Model([generator_input, identity_input], [discriminator_output, identity_output, forward_cycle, backward_cycle])
  #mse because the paper suggests using mse in place of log likelihood for adversarial loss
  #mae because paper uses L1 norm for identity and cyclic loss
  model.compile(loss=['mse', 'mae', 'mae', 'mae'], loss_weights=[1, 5, 10, 10], optimizer=keras.optimizers.Adam(lr=0.0002, beta_1=0.5))
  return model