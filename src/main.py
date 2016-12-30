from __future__ import print_function
import numpy as np
import tensorflow as tf
import time # For time stats
from six.moves import cPickle as pickle
from six.moves import range

TRAINING_ITERATIONS = 20000
BATCH_SIZE = 50
ACCURACY_TESTING_INTERVAL = 1000

IMAGE_SIZE = 28
NUM_LABELS = 10
TRAIN_SUBSET = 10000

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


print('\n************* Importing Data **************')

try:
  f = open('notMNIST.pickle', 'rb')
  saved_pickle = pickle.load(f)
  f.close()
except Exception as e:
  print('Unable to laod data from notMNIST.pickle:', e)
  raise

train_dataset = saved_pickle['train_dataset']
train_labels = saved_pickle['train_labels']
valid_dataset = saved_pickle['valid_dataset']
valid_labels = saved_pickle['valid_labels']
test_dataset = saved_pickle['test_dataset']
test_labels = saved_pickle['test_labels']
del saved_pickle  # Frees up memory

print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)
print('************* Data IBATCH_SIZEmported **************\n')

print('\n************* Reshaping Data **************\n')
def reformat(dataset, labels):
  dataset = dataset.reshape((-1, IMAGE_SIZE * IMAGE_SIZE)).astype(np.float32)
  # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
  labels = (np.arange(NUM_LABELS) == labels[:,None]).astype(np.float32)
  return dataset, labels
  
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)

print('\n************* After Reshaping Data **************\n')
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)
print('\n')


""" START SESSION """
sess = tf.InteractiveSession()

""" DEFINE VARIABLES WITH PLACEHOLDERS """
x = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE * IMAGE_SIZE])
y_ = tf.placeholder(tf.float32, shape=[None, NUM_LABELS])

""" INPUT """
x_image = tf.reshape(x, [-1, IMAGE_SIZE, IMAGE_SIZE, 1])	# Size = 1@28x28

""" LAYER 1 """
W_conv1 = weight_variable([5, 5, 1, 32])			# 5x5 Convolution
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)	# Output Size = 32@28x28
h_pool1 = max_pool_2x2(h_conv1)					# Output Size = 32@14x14

""" LAYER 2 """
W_conv2 = weight_variable([5, 5, 32, 64])			# 5x5 Convolution
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)	# Output Size = 64@14x14
h_pool2 = max_pool_2x2(h_conv2)					# Output Size = 64@7x7

""" FULLY CONNECTED LAYER """
W_fc1 = weight_variable([7 * 7 * 64, 256])			# Must match 64@7x7 Size
b_fc1 = bias_variable([256])					# FC Neural Net = 256 Neurons
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])		# Must match 64@7x7 Size
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

""" DROPOUT LAYER """
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

""" READOUT LAYER """
W_fc2 = weight_variable([256, 10])				# Output Needs 10 Classes
b_fc2 = bias_variable([10])

""" OUTPUT """
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# Define the loss function or cross entropy
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
# Define model training
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# Returns "1" if argmax(y_conv,1) = argmax(y_,1), otherwise returns "0"
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
# Casts the tensor to a "float" and then calculates the mean value
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initialise all the variables
print('\nTRAINING_ITERATIONS: ' + str(TRAINING_ITERATIONS))
print('BATCH_SIZE: ' + str(BATCH_SIZE))
sess.run(tf.initialize_all_variables())

print("Beginning training...\n")
# Start the timer Krunk!
start_time = time.time()

for i in range(TRAINING_ITERATIONS):
  start_index = (i * BATCH_SIZE) % train_dataset.shape[0]
  finish_index = start_index + BATCH_SIZE
  train_batch = [train_dataset[start_index:finish_2index, :], train_labels[start_index:finish_index, :]]
 
  if i % ACCURACY_TESTING_INTERVAL == 0:
    start_index = (i * BATCH_SIZE) % valid_dataset.shape[0]
    finish_index = start_index + BATCH_SIZE
    valid_batch = [valid_dataset[start_index:finish_index, :], valid_labels[start_index:finish_index, :]]
    train_accuracy = float(accuracy.eval(feed_dict={x: valid_batch[0], y_: valid_batch[1], keep_prob: 1.0}))
    print('Training Accuracy (Step {:0d}): {:.3f}%'.format(i, 100*train_accuracy))
    
  train_step.run(feed_dict={x: train_batch[0], y_: train_batch[1], keep_prob: 0.5})


# Finally print out the results
final_accuracy = []
for i in range(int(test_dataset.shape[0]/BATCH_SIZE)):

  start_index = (i * BATCH_SIZE) % test_dataset.shape[0]
  finish_index = start_index + BATCH_SIZE
  test_batch = [test_dataset[start_index:finish_index, :], test_labels[start_index:finish_index, :]]
  final_accuracy.append(accuracy.eval(feed_dict={x: test_batch[0], y_: test_batch[1], keep_prob: 1.0}))

print('\n*************************************************')
print('Final Model Accuracy: {:.2f}% (Runtime: {:.2f} Seconds)\n'.format(100*(sum(final_accuracy) / float(len(final_accuracy))), (time.time() - start_time)) )




