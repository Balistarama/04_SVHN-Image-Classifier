import sys

''' DATASET SETTINGS '''
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

''' TRAINING SIZES '''
# Number of times the network should train on ALL the training examples (effects training time!!! double X = 2.1x the time)
TRAINING_EPOCHS = 5

# The number of images and labels that get fed into the Neural Net each training iteration
BATCH_SIZE = 128

# After this many traininig iterations the training pauses and 
# the current training accuracy is tested using validation data
ACCURACY_TESTING_INTERVAL = 2000

''' MODEL DESIGN '''
# The size of the convolution patch (effects memory not training time)
CONVOLUTION_SIZE = 3								#int(sys.argv[1])

# Number of feature maps in the first layer (effects training time!!! double X = 1.8x the time)
LAYER_1_FEATURE_MAPS = 32

# Number of feature maps in the second layer
LAYER_2_FEATURE_MAPS = LAYER_1_FEATURE_MAPS * 2

# Number of feature maps in the third layer
LAYER_3_FEATURE_MAPS = LAYER_2_FEATURE_MAPS * 2

# Number of feature maps in the forth layer
LAYER_4_FEATURE_MAPS = LAYER_3_FEATURE_MAPS * 2

# Number of Neurons in the first, fully connected layer
LAYER_1_FC_NEURONS = 1024

# Number of Neurons in the second, fully connected layer
LAYER_2_FC_NEURONS = 256

''' MODEL HYPERPARAMETERS '''
# The learning rate for the gradient optimizer
LEARNING_RATE = 0.00001

# The Keep Probability used for the training steps ONLY
TRAINING_KEEP_PROB = 0.5

def print_configuration():
  """ A quick function to print out all the Neural Nets current settings """
  print('*************************************************')
  print('TRAINING_EPOCHS: {:,d}'.format(int(TRAINING_EPOCHS))) 
  print('BATCH_SIZE: {:,d}'.format(BATCH_SIZE))
  print('CONVOLUTION_SIZE: {:,d}'.format(CONVOLUTION_SIZE))
  print('FEATURE_MAPS: Layer 1 = {:,d}, Layer 2 = {:,d}, Layer 3 = {:,d}, Layer 4 = {:,d}'.format(LAYER_1_FEATURE_MAPS, LAYER_2_FEATURE_MAPS, LAYER_3_FEATURE_MAPS, LAYER_4_FEATURE_MAPS))
  print('FC_NEURONS: Layer 1 = {:,d}, Layer 2 = {:,d}'.format(LAYER_1_FC_NEURONS, LAYER_2_FC_NEURONS))
  print('LEARNING_RATE: {:f}'.format(LEARNING_RATE))
  print('TRAINING_KEEP_PROB: {:f}'.format(TRAINING_KEEP_PROB))
