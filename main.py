from utilities import *
from models import *
from train import *

import os
import random
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa


path = './'

horse_array_filename = path + 'horse_images_array'
zebra_array_filename = path + 'zebra_images_array'
horse_images = np.load(horse_array_filename+'.npy')
zebra_images = np.load(zebra_array_filename+'.npy')
print("Number of horse images: {}. Number of zebra images: {}.".format(horse_images.shape, zebra_images.shape))


image_shape = (256,256,3)

generator_horse_to_zebra = generator_model(image_shape)
generator_zebra_to_horse = generator_model(image_shape)

discriminator_horse = discriminator_model(image_shape)
discriminator_zebra = discriminator_model(image_shape)

update_generator_horse_to_zebra = generator_update_model(generator_horse_to_zebra, 
														discriminator_zebra, 
														generator_zebra_to_horse, 
														image_shape)

update_generator_zebra_to_horse = generator_update_model(generator_zebra_to_horse, 
														discriminator_horse, 
														generator_horse_to_zebra, 
														image_shape)

if __name__ == '__main__':
    train(discriminator_horse, discriminator_zebra, 
          generator_horse_to_zebra, generator_zebra_to_horse, 
          update_generator_horse_to_zebra, update_generator_zebra_to_horse, 
          horse_images, zebra_images)