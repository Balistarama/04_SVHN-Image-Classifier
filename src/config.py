# The number of training iterations the Neural Net goes through
TRAINING_ITERATIONS = 10000

# The number of images and labels that get fed into the Neural 
# Net each training iteration
BATCH_SIZE = 100

# After this many traininig iterations the training pauses and 
# the current training accuracy is tested using validation data
ACCURACY_TESTING_INTERVAL = 1000

# The size of the convolution patch
CONVOLUTION_SIZE = 3

# Number of Neurons in the final, fully connected layer
NUM_NEURONS = 128

# The size of the inputted images (assumes image is always a square)0
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

def print_configuration():
  """ A quick function to print out all the Neural Nets current settings """
  print('\nCurrent Configuration:')
  print('TRAINING_ITERATIONS: ' + '{:,d}'.format(TRAINING_ITERATIONS))
  print('BATCH_SIZE: ' + '{:,d}'.format(BATCH_SIZE))
  print('ACCURACY_TESTING_INTERVAL: ' '{:,d}'.format(ACCURACY_TESTING_INTERVAL))
  print('DATASET: ' + DATASET)
