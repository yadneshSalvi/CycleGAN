import random
import numpy as np

#the following function selects n random images from real dataset
def get_real_images(real_dataset, num_images, n_patch):
	random_indices = []
	while len(random_indices)!=num_images:
		ind = random.randint(0,real_dataset.shape[0])
		if ind not in random_indices:
			random_indices.append(ind)

	random_indices = np.array(random_indices)
	X = real_dataset[random_indices]
	y = np.ones((num_images, n_patch, n_patch, 1))
	return X, y

#the following function generates n fake images
def get_fake_images(generator, real_dataset, n_patch):
	X = generator.predict(real_dataset)
	y = np.zeros((len(X), n_patch, n_patch, 1))
	return X, y

#we maintain a pool of 50 fake generated images as mentioned in paer
#we keep on updating that pool of fake images and drawing new set of fake images from there
def fake_images_pool_update(fake_images_pool, generated_fake_images):
	x_fake = []
	for fake_image in generated_fake_images:
		if len(fake_images_pool) < 50:#pool size 50 is specified in paper
			fake_images_pool.append(fake_image)#if pool is not full add images
			x_fake.append(fake_image)
		elif random.uniform(0,1) < 0.5:# with prob 0.5 use this image but don't add to pool
			x_fake.append(fake_image)
		else:
			ind = random.randint(0, len(pool))#with prob 0.5 add this image in place of some random image and use that replaced random image
			x_fake.append(fake_images_pool[ind])
			fake_images_pool[ind] = fake_image
	x_fake = np.asarray(x_fake)
	return x_fake


def save_models(step, g_model_AtoB, g_model_BtoA):
	# save the first generator model
	filename1 = 'g_model_AtoB_%06d.h5' % (step+1)
	g_model_AtoB.save(filename1)
	# save the second generator model
	filename2 = 'g_model_BtoA_%06d.h5' % (step+1)
	g_model_BtoA.save(filename2)
	print('>Saved: %s and %s' % (filename1, filename2))