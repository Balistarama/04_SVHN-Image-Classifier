from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
from six.moves import urllib
from scipy.io import loadmat
import h5py
import random

NUM_LABELS = 10
IMAGE_SIZE = 32
NUM_CHANNELS = 3

# Load the raw Matlab files
train_f = loadmat("raw/Format 2/train_32x32.mat")
extra_f = loadmat("raw/Format 2/extra_32x32.mat")
test_f = loadmat("raw/Format 2/test_32x32.mat")

# Import the train and extra images and their associated labels
# then combine them together to make one single huge dataset
trainx = train_f['X']					# rows of the images (32x32x3)
trainy = train_f['y'].flatten()				# rows of the label values
extrax = extra_f['X']					# rows of the images (32x32x3)
extray = extra_f['y'].flatten()				# rows of the label values
train_dataset = np.concatenate((trainx[...], extrax[...]), axis=3)
train_labels  = np.concatenate((trainy, extray))
print('Training Set', train_dataset.shape, train_labels.shape)

# Import the test images and their associated labels
test_dataset = test_f['X']					# rows of the images (32x32x3)
test_labels = test_f['y'].flatten()				# rows of the label values

# Cut the test_dataset in half and use the second half as the validation dataset
half_way_point = int(test_labels.shape[0] / 2)
valid_dataset = test_dataset[:,:,:,half_way_point:test_dataset.shape[3]]
valid_labels = test_labels[half_way_point:test_labels.shape[0]]
test_dataset = test_dataset[:,:,:,0:half_way_point]
test_labels = test_labels[0:half_way_point]
print('Validation Set', valid_dataset.shape, valid_labels.shape)
print('Test Set', test_dataset.shape, test_labels.shape)

# Reformat into a TensorFlow-friendly shape:
# - convolutions need the image data formatted as a cube (width by height by #channels)
# - labels as float 1-hot encodings.
print('Reshaping Data...')
def reformat(dataset, labels):

  # Map (NUM_IMAGES, IMAGE_SIZE, IMAGE_SIZE) to (NUM_IMAGES, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
  # (32, 32, 3, 604388) -> (604388, 32, 32, 3) 
  reshaped_dataset = np.ndarray((dataset.shape[3], IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS), dtype=np.float32)
  for i in range(dataset.shape[3]):
    reshaped_dataset[i,:,:,:] = dataset[:,:,:,i]

  # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
  reshaped_labels = (np.arange(NUM_LABELS) == labels[:,None]).astype(np.int)
  
  return reshaped_dataset, reshaped_labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)

print('Data Reshaped!\n')
print('Training Set', train_dataset.shape, train_labels.shape)
print('Validation Set', valid_dataset.shape, valid_labels.shape)
print('Test Set', test_dataset.shape, test_labels.shape)

# Uncomment this section if you'd like to view a random image from each of the datasets
''' 
print('Train Image')
rand_image = random.randint(0, 10000)
print (rand_image)
plt.imshow(train_dataset[rand_image])
print (train_labels[rand_image])
plt.show()


print('Validation Image')
rand_image = random.randint(0, 10000)
print (rand_image)
plt.imshow(valid_dataset[rand_image])
print (valid_labels[rand_image])
plt.show()

print('Test Image')
rand_image = random.randint(0, 10000)
print (rand_image)
plt.imshow(test_dataset[rand_image])
print (test_labels[rand_image])
plt.show()
'''

# Finally, let's save the data for later reuse
pickle_file = 'SVHN.pickle'
print('Doing some pickling....')

try:
  f = open(pickle_file, 'wb')
  save = {
    'train_dataset': train_dataset,
    'train_labels': train_labels,
    'valid_dataset': valid_dataset,
    'valid_labels': valid_labels,
    'test_dataset': test_dataset,
    'test_labels': test_labels,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise
  
statinfo = os.stat(pickle_file)
print('Pickling completed. Compressed pickle size: ', statinfo.st_size)
