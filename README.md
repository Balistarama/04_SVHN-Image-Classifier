<div align="center">
  <img src="https://raw.githubusercontent.com/Balistarama/04_SVHN-Image-Classifier/master/Google%20Street%20View.png"><br>
</div>

##04_SVHN Image Classifier
Using the Street View House Numbers (SVHN) Dataset, this project is a state of
the art Deep Convolutional Neural Network image to number classifier.

This project is the final task in the Udacity Deep Learning course by Google 
that can be found here: https://www.udacity.com/course/deep-learning--ud730

##OBJECTIVE:
Build an application that can interpret numbers from real-world images.

##TODO:
- Build and configure data_processor.py to load the Matlab files, extract the 
images/labels and then pickle them into a single data file that the main.py 
code can use <img src="https://raw.githubusercontent.com/Balistarama/04_SVHN-Image-Classifier/master/tick.png">
- Update data_processor.py to split out a few thousand examples for the validation dataset <img src="https://raw.githubusercontent.com/Balistarama/04_SVHN-Image-Classifier/master/tick.png">
- Configure main.py to properly import the pickled file <img src="https://raw.githubusercontent.com/Balistarama/04_SVHN-Image-Classifier/master/tick.png">
- Update data_processor.py to reshape the data so it's the right formate for the Neural Net <img src="https://raw.githubusercontent.com/Balistarama/04_SVHN-Image-Classifier/master/tick.png">
- Setup the Neural Net so that it accepts the new image tensors <img src="https://raw.githubusercontent.com/Balistarama/04_SVHN-Image-Classifier/master/tick.png">
- Setup TensorBoard
- Increase accuracy, current value: 16.13%

##INSTRUCTIONS:
 - Clone repository
 - Follow the data/README to download and munge the raw dataset
 - Run the src/main.py Neural Network

##REFERENCES:
You're welcome to use any and all code but a reference back to 
https://github.com/Balistarama would be appreciated
