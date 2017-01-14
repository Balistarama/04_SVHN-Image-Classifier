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
TRAINING_EPOCHS = 100

# The number of images and labels that get fed into the Neural Net each training iteration
BATCH_SIZE = 128

# The number of training iterations the Neural Net goes through
TRAINING_ITERATIONS = int(TRAINING_EPOCHS * (604388/BATCH_SIZE))

# After this many traininig iterations the training pauses and 
# the current training accuracy is tested using validation data
ACCURACY_TESTING_INTERVAL = 50

''' MODEL DESIGN '''
# The size of the convolution patch (effects memory not training time)
CONVOLUTION_SIZE = 5

# Number of feature maps in the first layer (effects training time!!! double X = 1.8x the time)
LAYER_1_FEATURE_MAPS = 8

# Number of feature maps in the second layer
LAYER_2_FEATURE_MAPS = LAYER_1_FEATURE_MAPS * 2

# Number of feature maps in the third layer
LAYER_3_FEATURE_MAPS = LAYER_2_FEATURE_MAPS * 2

# Number of Neurons in the first, fully connected layer
LAYER_1_FC_NEURONS = 4096

''' MODEL HYPERPARAMETERS '''
# The learning rate for the gradient optimizer
LEARNING_RATE = 0.00001

# The Keep Probability used for the training steps ONLY
TRAINING_KEEP_PROB = 0.9

def print_configuration():
  """ A quick function to print out all the Neural Nets current settings """
  print('\n*************************************************')
  print('TRAINING_EPOCHS: {:,d}'.format(int(TRAINING_EPOCHS))) 
  print('BATCH_SIZE: {:,d}'.format(BATCH_SIZE))
  print('TRAINING_ITERATIONS: {:,d}'.format(TRAINING_ITERATIONS))
  print('CONVOLUTION_SIZE: {:,d}'.format(CONVOLUTION_SIZE))
  print('FEATURE_MAPS: Layer 1 = {:,d}, Layer 2 = {:,d}, Layer 3 = {:,d}'.format(LAYER_1_FEATURE_MAPS, LAYER_2_FEATURE_MAPS, LAYER_3_FEATURE_MAPS))
  print('LAYER_1_FC_NEURONS: {:,d}'.format(LAYER_1_FC_NEURONS))
  print('LEARNING_RATE: {:f}'.format(LEARNING_RATE))
  print('TRAINING_KEEP_PROB: {:f}'.format(TRAINING_KEEP_PROB))
