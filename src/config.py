''' Settings regarding the dataset and image properties '''
# The size of the inputted images (assumes image is always a square)
IMAGE_SIZE = 32

# The number of channels for each image (eg. RGB = 3)
NUM_CHANNELS = 3

# The pixel depth of the imgae
PIXEL_DEPTH = 255

# The number of classes for the system to classigy
NUM_LABELS = 10

# Path the TensorBoard logs are all saved to
LOGS_PATH = "../logs"

# Location of preprocessed dataset, see data/README for processing details
DATASET = '../data/SVHN.pickle'

''' Model Hyperparamaters '''
# Number of times the network should train on ALL the training examples
TRAINING_EPOCHS = 1

# The number of images and labels that get fed into the Neural 
# Net each training iteration
BATCH_SIZE = 50

# The number of training iterations the Neural Net goes through
TRAINING_ITERATIONS = TRAINING_EPOCHS * int(604388/BATCH_SIZE)

# After this many traininig iterations the training pauses and 
# the current training accuracy is tested using validation data
ACCURACY_TESTING_INTERVAL = 50

# The size of the convolution patch
CONVOLUTION_SIZE = 5

# Number of feature maps in the first layer
LAYER_1_FEATURE_MAPS = 4

# Number of feature maps in the second layer
LAYER_2_FEATURE_MAPS = LAYER_1_FEATURE_MAPS * 2

# Number of Neurons in the final, fully connected layer
NUM_NEURONS = 128

# The learning rate for the gradient optimizer
LEARNING_RATE = 0.00001

# The Keep Probability used for the training steps ONLY
TRAINING_KEEP_PROB = 0.9

def print_configuration():
  """ A quick function to print out all the Neural Nets current settings """
  print('Current Configuration:')
  print('TRAINING_ITERATIONS: ' + '{:,d}'.format(TRAINING_ITERATIONS))
  print('BATCH_SIZE: ' + '{:,d}'.format(BATCH_SIZE))
  print('ACCURACY_TESTING_INTERVAL: ' '{:,d}'.format(ACCURACY_TESTING_INTERVAL))
  print('CONVOLUTION_SIZE: ' '{:,d}'.format(CONVOLUTION_SIZE))
  print('NUM_NEURONS: ' '{:,d}'.format(NUM_NEURONS))
  print('LEARNING_RATE: ' '{:f}'.format(LEARNING_RATE))
  print('LAYER_1_FEATURE_MAPS: ' '{:,d}'.format(LAYER_1_FEATURE_MAPS))
  print('LAYER_2_FEATURE_MAPS: ' '{:,d}'.format(LAYER_2_FEATURE_MAPS)) 
  print('TRAINING_KEEP_PROB: ' '{:f}'.format(TRAINING_KEEP_PROB))
  print('TRAINING_EPOCHS: ' '{:,d}'.format(TRAINING_EPOCHS)) 
  print('DATASET: ' + DATASET)
