<div align="center">
  <img src="https://github.com/Balistarama/04_SVHN-Image-Classifier/blob/master/images/Google%20Street%20View.png?raw=true"><br>
</div>

#04_SVHN Image Classifier
Using the Street View House Numbers (SVHN) Dataset (Format 2), this project implements a 
Deep Convolutional Neural Network to perform image to number classiffication.

This project is also the final task in the Udacity Deep Learning course by Google 
that can be found here: https://www.udacity.com/course/deep-learning--ud730

##OBJECTIVE:
Build an application that can interpret numbers from real-world images.

##CURRENT BEST ACCURACY ACHIEVED:
- 88.68% (Runtime: 646.84 Minutes)

##SYSTEM USED:
###Software
- Ubunutu
- TensorFlow
- numpy
- six.moves
###Hardware
- CPU: Intel Core i7-6700 3.4GHz Quad-Core Processor
- Motherboard: Gigabyte GA-Z170X-Gaming 7 ATX LGA1151 Motherboard
- Memory: Corsair Vengeance LPX 32GB (4 x 8GB) DDR4-2666 Memory
- Storage: Samsung 960 PRO 512GB M.2-2280 Solid State Drive
- Video Card: Gigabyte GeForce GTX 1070 8GB G1 Gaming Video Card

##TODO:
- Build and configure data_processor.py to load the Matlab files, extract the 
images/labels and then pickle them into a single data file that the main.py 
code can use <img src="https://github.com/Balistarama/04_SVHN-Image-Classifier/blob/master/images/tick.png?raw=true">
- Update data_processor.py to split out a few thousand examples for the validation dataset <img src="https://github.com/Balistarama/04_SVHN-Image-Classifier/blob/master/images/tick.png?raw=true">
- Configure main.py to properly import the pickled file <img src="https://github.com/Balistarama/04_SVHN-Image-Classifier/blob/master/images/tick.png?raw=true">
- Update data_processor.py to reshape the data so it's the right formate for the Neural Net <img src="https://github.com/Balistarama/04_SVHN-Image-Classifier/blob/master/images/tick.png?raw=true">
- Setup the Neural Net so that it accepts the new image tensors <img src="https://github.com/Balistarama/04_SVHN-Image-Classifier/blob/master/images/tick.png?raw=true">
- Setup TensorBoard <img src="https://github.com/Balistarama/04_SVHN-Image-Classifier/blob/master/images/tick.png?raw=true">
- Fix up whatever the hell is wrong with the data processing as it's outputting rubbish!!! <img src="https://github.com/Balistarama/04_SVHN-Image-Classifier/blob/master/images/tick.png?raw=true">
- Redefined all the hyperparamaters to be in the config.py file <img src="https://github.com/Balistarama/04_SVHN-Image-Classifier/blob/master/images/tick.png?raw=true">
- Increase the models accuracy...
- Once model accuracy is high enough setup [TPOT](https://github.com/rhiever/tpot) and confirm that it's running at peak efficiency
- Deploy the final solution to the Google Machine Learning Cloud
- Write up report

##INSTRUCTIONS
###data/data_processor.py
This file takes in the raw .mat files that are downloaded from the Stanford website and munges it into a form that is both accepted by the network but also splits and joins things. The original data sets come in three "train, test and extra" .mat files.
- http://ufldl.stanford.edu/housenumbers/train_32x32.mat
- http://ufldl.stanford.edu/housenumbers/test_32x32.mat
- http://ufldl.stanford.edu/housenumbers/extra_32x32.mat
This file merges the two "train" and "extra" files to create a huge training set:
Training Set:  (604388, 32, 32, 3) (604388, 10)

It then splits up the "test" images into two separate "validation" and "testing" datasets:
Validation Set:  (13016, 32, 32, 3) (13016, 10)
Testing Set: (13016, 32, 32, 3) (13016, 10)

Finally it reshapes them all and writes them to a single pickle file. You will need to download the .mat files and then run this data_processor.py script ONCE to create the .pickle file.
```
balistarama@Computer:~/04_SVHN Image Classifier/data$ python3 data_processor.py
```

###src/config.py
This file is only used to configure the neural network that's stored in src/main.py. Open the file up in any text editor type in the batch size, number of feature maps on each layer and many more paramaters (including hyperaramters too) and then run the main.py file.

###src/main.py
This file is where the network is built, defined and trained. Simply setup your configuration setting in the config.py file and run main.py!
```
balistarama@Computer:~/04_SVHN Image Classifier/src$ python3 main.py
```
It will let you know what's happening at every step of the way including printing out the sizes of the datasets being used for the training as well as giving you feedback as the network is trained.
```
Training Accuracy: 77% - Iteration: 5,978,000/6,043,880 (98%) - Time Remaining: 7.05 Minutes
Training Accuracy: 93% - Iteration: 5,980,000/6,043,880 (98%) - Time Remaining: 6.84 Minutes
Training Accuracy: 82% - Iteration: 5,982,000/6,043,880 (98%) - Time Remaining: 6.62 Minutes
Training Accuracy: 85% - Iteration: 5,984,000/6,043,880 (99%) - Time Remaining: 6.41 Minutes
Training Accuracy: 86% - Iteration: 5,986,000/6,043,880 (99%) - Time Remaining: 6.20 Minutes
Training Accuracy: 87% - Iteration: 5,988,000/6,043,880 (99%) - Time Remaining: 5.98 Minutes
...
```

##TENSORBOARD:
In a seperate terminal window and from the main directory run:
```
tensorboard --logdir=./logs
```

##REFERENCES:
You're welcome to use any and all code but a reference back to 
https://github.com/Balistarama would be appreciated

