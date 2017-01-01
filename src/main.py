from __future__ import print_function
import numpy as np
import tensorflow as tf
import time # For time stats
from six.moves import cPickle as pickle
from six.moves import range
from config import *

def weight_variable(shape):
  """ Initialize the weights with a small amount of noise for symmetry breaking """
  """ and to prevent 0 gradients """
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  """ Since we're using ReLU neurons, it's good practice to initialize them with a """
  """ slightly positive initial bias to avoid "dead neurons" """
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  """ Computes a 2-D convolution given 4-D input and filter tensors """
  """ strides: The stride of the sliding window for each dimension of input """
  """ Our convolutions has the OUTPUT THE SAME SIZE AS THE INPUT. """  
  return tf.nn.conv2d(input=x, filter=W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  """ Performs max pooling on the input. """
  """ value: A 4-D Tensor with shape [batch, height, width, channels] """
  """ ksize: The size of the window for each dimension of the input tensor. """
  """ strides: The stride of the sliding window for each dimension of the input tensor. """
  return tf.nn.max_pool(value=x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Import the dataset from the main pickle file
print('\nImporting Data from "' + DATASET + '"')

try:
  f = open(DATASET, 'rb')
  saved_pickle = pickle.load(f)
  f.close()
except Exception as e:
  print('Unable to laod data from ', DATASET, ':', e)
  raise

train_dataset = saved_pickle['train_dataset']
train_labels = saved_pickle['train_labels']
valid_dataset = saved_pickle['valid_dataset']
valid_labels = saved_pickle['valid_labels']
test_dataset = saved_pickle['test_dataset']
test_labels = saved_pickle['test_labels']
del saved_pickle  # Frees up memory

print('Data Imported!')
print('Training Set: ', train_dataset.shape, train_labels.shape)
print('Validation Set: ', valid_dataset.shape, valid_labels.shape)
print('Testing Set:', test_dataset.shape, test_labels.shape)
print('\n')

# Define the entire model of the Neural Network for Tensorflow
""" START SESSION """
sess = tf.InteractiveSession()

""" DEFINE VARIABLES WITH PLACEHOLDERS """
with tf.name_scope('Input_Images'):
    x = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS], name="x-input")

with tf.name_scope('Input_Labels'):
    y_ = tf.placeholder(tf.float32, shape=[None, NUM_LABELS], name="y-input")

""" LAYER 1 """
with tf.name_scope('Layer_1_Weights'):
    W_conv1 = weight_variable([CONVOLUTION_SIZE, CONVOLUTION_SIZE, NUM_CHANNELS, 28])

with tf.name_scope('Layer_1_Biases'):    
    b_conv1 = bias_variable([28])

h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)							# Output Size = 32@32x32
h_pool1 = max_pool_2x2(h_conv1)										# Output Size = 32@16x16

""" LAYER 2 """
with tf.name_scope('Layer_2_Weights'):
    W_conv2 = weight_variable([CONVOLUTION_SIZE, CONVOLUTION_SIZE, 28, 56])

with tf.name_scope('Layer_2_Biases'):
    b_conv2 = bias_variable([56])
    
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)						# Output Size = 64@16x16
h_pool2 = max_pool_2x2(h_conv2)										# Output Size = 64@8x8

""" FULLY CONNECTED LAYER """
with tf.name_scope('FCL_1_Weights'):
    W_fc1 = weight_variable([8 * 8 * 56, NUM_NEURONS])	

with tf.name_scope('FCL_1_Biases'):						# Must match 64@8x8 Size
    b_fc1 = bias_variable([NUM_NEURONS])

h_pool2_flat = tf.reshape(h_pool2, [-1, 8 * 8 * 56])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

""" DROPOUT LAYER """
with tf.name_scope('Dropout'):
    keep_prob = tf.placeholder(tf.float32)
    tf.summary.scalar('dropout_keep_probability', keep_prob)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

""" READOUT LAYER """
with tf.name_scope('FCL_2_Weights'):
    W_fc2 = weight_variable([NUM_NEURONS, NUM_LABELS])
    
with tf.name_scope('FCL_2_Biases'):
    b_fc2 = bias_variable([NUM_LABELS])

""" OUTPUT """
with tf.name_scope("Softmax"):
    y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

""" LOSS FUNCTION """
# Define the loss function or cross entropy
with tf.name_scope('Cross_Entropy'):
    with tf.name_scope('Total'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
tf.summary.scalar("Cross_Entropy", cross_entropy)
	
# Define model training
with tf.name_scope('Train'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        
with tf.name_scope('Accuracy'):
    with tf.name_scope('Correct_Prediction'):
        # Returns "1" if argmax(y_conv,1) = argmax(y_,1), otherwise returns "0"
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    with tf.name_scope('Accuracy'):
        # Casts the tensor to a "float" and then calculates the mean value
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar("Accuracy", accuracy)    

# Merge all the summaries and write them out 
merged = tf.summary.merge_all()
validation_writer = tf.summary.FileWriter(LOGS_PATH)
train_writer = tf.summary.FileWriter(LOGS_PATH)

# Initialise all the variables
tf.global_variables_initializer().run()

print_configuration()

print("Beginning training...\n")
# Start the timer Krunk!
start_time = time.time()

for i in range(TRAINING_ITERATIONS):
 
  if i % ACCURACY_TESTING_INTERVAL == 0:
    start_index = (i * BATCH_SIZE) % valid_dataset.shape[0]
    finish_index = start_index + BATCH_SIZE
    valid_batch = [valid_dataset[start_index:finish_index, :], valid_labels[start_index:finish_index, :]]
    validation_accuracy = accuracy.eval(feed_dict={x: valid_batch[0], y_: valid_batch[1], keep_prob: 1.0})
    #print('Training Accuracy (Step {:0d}): {:.3f}%'.format(i, 100 * validation_accuracy))
    #validation_writer.add_summary(validation_accuracy, i)

  start_index = (i * BATCH_SIZE) % train_dataset.shape[0]
  finish_index = start_index + BATCH_SIZE
  train_batch = [train_dataset[start_index:finish_index, :], train_labels[start_index:finish_index, :]]    
  train_accuracy = train_step.run(feed_dict={x: train_batch[0], y_: train_batch[1], keep_prob: 0.5})
  train_writer.add_summary(train_accuracy, i)

# Finally test and print out the results
final_accuracy = []
for i in range(int(test_dataset.shape[0]/BATCH_SIZE)):

  start_index = (i * BATCH_SIZE) % test_dataset.shape[0]
  finish_index = start_index + BATCH_SIZE
  test_batch = [test_dataset[start_index:finish_index, :], test_labels[start_index:finish_index, :]]
  final_accuracy.append(accuracy.eval(feed_dict={x: test_batch[0], y_: test_batch[1], keep_prob: 1.0}))

print('\n*************************************************')
print('Final Model Accuracy: {:.2f}% (Runtime: {:.2f} Seconds)\n'.format(100*(sum(final_accuracy) / float(len(final_accuracy))), (time.time() - start_time)) )





