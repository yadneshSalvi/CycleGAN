import os
import numpy as np

def get_image_dataset(dataset_path):
  img_dataset = []
  for image in os.listdir(dataset_path):
    img = keras.preprocessing.image.load_img(os.path.join(dataset_path,image))
    img = keras.preprocessing.image.img_to_array(img)
    img_dataset.append(img)
  return np.array(img_dataset)

path = './'

horse_train_images = get_image_dataset(path + 'trainA/')
horse_test_images = get_image_dataset(path + 'testA/')
# we do not want seperate train and test images so we will stack them one after the other
horse_images = np.vstack((horse_train_images, horse_test_images))

zebra_train_images = get_image_dataset(path + 'trainB/')
zebra_test_images = get_image_dataset(path + 'testB/')
zebra_images = np.vstack((zebra_train_images, zebra_test_images))

print("Number of horse images: {}. Number of zebra images: {}.".format(horse_images.shape, zebra_images.shape))

#plot the first five images
for img_num in range(1,6):
  plt.subplot(2,5,img_num); plt.imshow(horse_images[img_num].astype('uint8'))
  plt.subplot(2,5,5+img_num); plt.imshow(zebra_images[img_num].astype('uint8'))

# it takes a lot of time to create the image arrays so we will save these arrays for further use and load them directly
horse_array_filename = path + 'horse_images_array'
zebra_array_filename = path + 'zebra_images_array'
np.save(horse_array_filename, horse_images)
np.save(zebra_array_filename, zebra_images)

# now lets load the above saved array and plot them to see if they were saved correctly
horse_images = np.load(horse_array_filename+'.npy')
zebra_images = np.load(zebra_array_filename+'.npy')
print("Number of horse images: {}. Number of zebra images: {}.".format(horse_images.shape, zebra_images.shape))

# plot some of the images
for img_num in range(1,6):
  plt.subplot(2,5,img_num); plt.imshow(horse_images[img_num].astype('uint8'))
  plt.subplot(2,5,5+img_num); plt.imshow(zebra_images[img_num].astype('uint8'))