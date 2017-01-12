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
  #initial = tf.truncated_normal(shape, stddev=0.1)
  initial = tf.random_normal(shape, stddev=1.0)
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

def variable_summaries(var, name):
  """Automatically attaches a bunch of summaries to a Tensor for TensorBoard."""
  with tf.name_scope('Summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('Mean/' + name, mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev/' + name, stddev)
    tf.summary.scalar('max/' + name, tf.reduce_max(var))
    tf.summary.scalar('min/' + name, tf.reduce_min(var))
    tf.summary.histogram(name, var)

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
with tf.name_scope('Inputs'):
    x = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS], name="x-input")
    y_ = tf.placeholder(tf.float32, shape=[None, NUM_LABELS], name="y-input")
    tf.summary.image('input', x, 3)
	
""" LAYER 1 """
layer_name = 'Layer_1'
with tf.name_scope(layer_name):
    with tf.name_scope('Weights'):
        W_conv1 = weight_variable([CONVOLUTION_SIZE, CONVOLUTION_SIZE, NUM_CHANNELS, LAYER_1_FEATURE_MAPS])
        variable_summaries(W_conv1, layer_name + '/Weights')

    with tf.name_scope('Biases'):
        b_conv1 = bias_variable([LAYER_1_FEATURE_MAPS])
        variable_summaries(b_conv1, layer_name + '/Biases')    
    
    with tf.name_scope('Wx_plus_b'):
        preactivate = conv2d(x, W_conv1) + b_conv1
        tf.summary.histogram(layer_name + '/pre_activations', preactivate)
        activations = tf.nn.relu(preactivate, name='activation')
        tf.summary.histogram(layer_name + '/activations', activations)  
        h_pool1 = max_pool_2x2(activations)
        tf.summary.histogram(layer_name + '/poolings', h_pool1)  
        
""" LAYER 2 """
layer_name = 'Layer_2'
with tf.name_scope(layer_name):
    with tf.name_scope('Weights'):
        W_conv2 = weight_variable([CONVOLUTION_SIZE, CONVOLUTION_SIZE, LAYER_1_FEATURE_MAPS, LAYER_2_FEATURE_MAPS])
        variable_summaries(W_conv2, layer_name + '/Weights')

    with tf.name_scope('Biases'):
        b_conv2 = bias_variable([LAYER_2_FEATURE_MAPS])
        variable_summaries(b_conv2, layer_name + '/Biases')    
    
    with tf.name_scope('Wx_plus_b'):
        preactivate = conv2d(h_pool1, W_conv2) + b_conv2
        tf.summary.histogram(layer_name + '/pre_activations', preactivate)
        activations = tf.nn.relu(preactivate, name='activation')
        tf.summary.histogram(layer_name + '/activations', activations)  
        h_pool2 = max_pool_2x2(activations)
        tf.summary.histogram(layer_name + '/poolings', h_pool2)  

""" FULLY CONNECTED LAYER """
layer_name = 'FCL_1'
with tf.name_scope(layer_name):
    with tf.name_scope('Weights'):
        W_fc1 = weight_variable([8 * 8 * LAYER_2_FEATURE_MAPS, NUM_NEURONS])	
        variable_summaries(W_fc1, layer_name + '/Weights')

    with tf.name_scope('Biases'):
        b_fc1 = bias_variable([NUM_NEURONS])
        variable_summaries(b_fc1, layer_name + '/Biases')    
    
    h_pool2_flat = tf.reshape(h_pool2, [-1, 8 * 8 * LAYER_2_FEATURE_MAPS])
    
    with tf.name_scope('Wx_plus_b'):
        preactivate = tf.matmul(h_pool2_flat, W_fc1) + b_fc1
        tf.summary.histogram(layer_name + '/pre_activations', preactivate)
        h_fc1 = tf.nn.relu(preactivate, name='activation')
        tf.summary.histogram(layer_name + '/activations', h_fc1) 

""" DROPOUT LAYER """
with tf.name_scope('Dropout'):
    keep_prob = tf.placeholder(tf.float32)
    tf.summary.scalar('dropout_keep_probability', keep_prob)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

""" OUTPUT """
layer_name = 'Output_Layer'
with tf.name_scope(layer_name):
    with tf.name_scope('Weights'):
        W_fc2 = weight_variable([NUM_NEURONS, NUM_LABELS])
        variable_summaries(W_fc2, layer_name + '/Weights')

    with tf.name_scope('Biases'):
        b_fc2 = bias_variable([NUM_LABELS])
        variable_summaries(b_fc2, layer_name + '/Biases')  

    with tf.name_scope('Wx_plus_b'):
        preactivate = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        tf.summary.histogram(layer_name + '/pre_activations', preactivate)
        y = tf.nn.softmax(preactivate, name='activation')
        tf.summary.histogram(layer_name + '/activations', y) 

""" LOSS FUNCTION """
# Define the loss function or cross entropy
with tf.name_scope('Cross_Entropy'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
tf.summary.scalar("Cross_Entropy", cross_entropy)
	
# Define model training
with tf.name_scope('Train'):
    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)
        
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
validation_writer = tf.summary.FileWriter(LOGS_PATH, sess.graph)
train_writer = tf.summary.FileWriter(LOGS_PATH, sess.graph)

# Initialise all the variables
tf.global_variables_initializer().run()

print('\n')
print_configuration()
print("Beginning training...\n")
# Start the timer Krunk!
start_time = time.time()

for i in range(TRAINING_ITERATIONS):
 
  if i % ACCURACY_TESTING_INTERVAL == 0:
    start_index = (i * BATCH_SIZE) % valid_dataset.shape[0]
    finish_index = start_index + BATCH_SIZE
    valid_batch = [valid_dataset[start_index:finish_index, :], valid_labels[start_index:finish_index, :]]
    #validation_accuracy = accuracy.eval(feed_dict={x: valid_batch[0], y_: valid_batch[1], keep_prob: 1.0})
    summary, acc = sess.run([merged, accuracy], feed_dict={x: valid_batch[0], y_: valid_batch[1], keep_prob: 1.0})
    validation_writer.add_summary(summary, i)
    
    if i != 0:
      estimated_time_remaining = ( ((time.time() - start_time)/60) * (1 / (i/TRAINING_ITERATIONS)) ) - ( (time.time() - start_time)/60 )
      #print('Training Accuracy: %s%   / Iteration: %s/%s   / Time Remaining: %f' % (int(100*acc), i, ACCURACY_TESTING_INTERVAL, estimated_time_remaining))
      print('Training Accuracy: {:.0f}% - Iteration: {:,d}/{:,d} ({:.0f}%) - Time Remaining: {:.2f} Minutes'.format(int(100*acc), i, TRAINING_ITERATIONS, int(100*(i/TRAINING_ITERATIONS)), estimated_time_remaining))      

  start_index = (i * BATCH_SIZE) % train_dataset.shape[0]
  finish_index = start_index + BATCH_SIZE
  train_batch = [train_dataset[start_index:finish_index, :], train_labels[start_index:finish_index, :]]    
  #train_accuracy = train_step.run(feed_dict={x: train_batch[0], y_: train_batch[1], keep_prob: 0.5})
  summary, _ = sess.run([merged, train_step], feed_dict={x: train_batch[0], y_: train_batch[1], keep_prob: TRAINING_KEEP_PROB})
  train_writer.add_summary(summary, i)

# Finally test and print out the results
final_accuracy = []
for i in range(int(test_dataset.shape[0]/BATCH_SIZE)):

  start_index = (i * BATCH_SIZE) % test_dataset.shape[0]
  finish_index = start_index + BATCH_SIZE
  test_batch = [test_dataset[start_index:finish_index, :], test_labels[start_index:finish_index, :]]
  final_accuracy.append(accuracy.eval(feed_dict={x: test_batch[0], y_: test_batch[1], keep_prob: 1.0}))

validation_writer.close()
train_writer.close()

print('\n*************************************************')
print('Final Model Accuracy: {:.2f}% (Runtime: {:.2f} Minutes)\n'.format(100*(sum(final_accuracy) / float(len(final_accuracy))), (time.time() - start_time)/60) )
print_configuration()



