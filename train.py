from utilities import *
from models import *

def train(discriminator_horse, discriminator_zebra, 
          generator_horse_to_zebra, generator_zebra_to_horse, 
          update_generator_horse_to_zebra, update_generator_zebra_to_horse, 
          horse_images, zebra_images):
	
  n_epochs, n_batch, = 100, 1
  n_patch = discriminator_horse.output_shape[1]

  horse_images = horse_images
  zebra_images = zebra_images

  fake_images_pool_horse = []
  fake_images_pool_zebra = []

  batches_per_epoch = int(len(horse_images) / n_batch)

  n_steps = batches_per_epoch * n_epochs
  print("Number of training steps : {}".format(n_steps))

  for i in range(n_steps):
    horse_real_images, y_horse_real_images = get_real_images(horse_images, n_batch, n_patch)
    zebra_real_images, y_zebra_real_images = get_real_images(zebra_images, n_batch, n_patch)
    
    horse_fake_images, y_horse_fake_images = get_fake_images(generator_zebra_to_horse, zebra_real_images, n_patch)
    zebra_fake_images, y_zebra_fake_images = get_fake_images(generator_horse_to_zebra, horse_real_images, n_patch)
    
    horse_fake_images = fake_images_pool_update(fake_images_pool_horse, horse_fake_images)
    zebra_fake_images = fake_images_pool_update(fake_images_pool_zebra, zebra_fake_images)
    
    #first update the zebra generator
    zebra_generator_loss, l, m, n, o  = update_generator_zebra_to_horse.train_on_batch([zebra_real_images, horse_real_images], 
                                                                        [y_horse_real_images, horse_real_images, zebra_real_images, 
                                                                         horse_real_images])
    #then update horse discriminator
    horse_discriminator_loss_on_real = discriminator_zebra.train_on_batch(horse_real_images, y_horse_real_images)
    horse_discriminator_loss_on_fake = discriminator_zebra.train_on_batch(horse_fake_images, y_horse_fake_images)
    
    #now update horse generator
    horse_generator_loss, l, m, n, o = update_generator_horse_to_zebra.train_on_batch([horse_real_images, zebra_real_images], 
                                                                       [y_zebra_real_images, zebra_real_images, horse_real_images,
                                                                        zebra_real_images])
    #then update zebra discriminator
    zebra_discriminator_loss_on_real = discriminator_horse.train_on_batch(zebra_real_images, y_zebra_real_images)
    zebra_discriminator_loss_on_fake = discriminator_horse.train_on_batch(zebra_fake_images, y_zebra_fake_images)
    
    #print the model performance at each step
    print("Step {} \n Horse Discriminator loss on real {} loss on fake {} \n Zebra Discriminator loss on real {} loss on fake {} \n Horse generator loss {} \n Zebra generator loss {} \n".format(i+1, horse_discriminator_loss_on_real, horse_discriminator_loss_on_fake, zebra_discriminator_loss_on_real, zebra_discriminator_loss_on_fake, horse_generator_loss, zebra_generator_loss))
    if (i+1) % (batches_per_epoch * 5) == 0:
      # save the models
      save_models(i, g_model_AtoB, g_model_BtoA)