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


train_f = loadmat("raw/Format 2/train_32x32.mat")
extra_f = loadmat("raw/Format 2/extra_32x32.mat")
test_f = loadmat("raw/Format 2/test_32x32.mat")

trainx = train_f['X']					# rows of the images (32x32x3)
trainy = train_f['y'].flatten()				# rows of the label values
extrax = extra_f['X']					# rows of the images (32x32x3)
extray = extra_f['y'].flatten()				# rows of the label values
train_dataset = np.concatenate((trainx[...,np.newaxis], extrax[...,np.newaxis]), axis=3)
train_labels  = np.concatenate((trainy, extray))
print('train_dataset shape: ' + str(train_dataset.shape))
print('train_labels shape: ' + str(train_labels.shape))

test_dataset = test_f['X']					# rows of the images (32x32x3)
test_labels = test_f['y'].flatten()				# rows of the label values
print('test_dataset shape: ' + str(test_dataset.shape))
print('test_labels shape: ' + str(test_labels.shape))

#valid_dataset, valid_labels

""" Finally, let's save the data for later reuse """
pickle_file = 'SVHN.pickle'
print('Doing some pickling....')

try:
  f = open(pickle_file, 'wb')
  save = {
    'train_dataset': train_dataset,
    'train_labels': train_labels,
#    'valid_dataset': valid_dataset,
#    'valid_labels': valid_labels,
    'test_dataset': test_dataset,
    'test_labels': test_labels,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise
  
statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)
