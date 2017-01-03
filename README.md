<div align="center">
  <img src="https://github.com/Balistarama/04_SVHN-Image-Classifier/blob/master/images/Google%20Street%20View.png?raw=true"><br>
</div>

##04_SVHN Image Classifier
Using the Street View House Numbers (SVHN) Dataset (Format 2), this project is a 
Deep Convolutional Neural Network image to number classifier.

This project is also the final task in the Udacity Deep Learning course by Google 
that can be found here: https://www.udacity.com/course/deep-learning--ud730

##OBJECTIVE:
Build an application that can interpret numbers from real-world images.

##TODO:
- Build and configure data_processor.py to load the Matlab files, extract the 
images/labels and then pickle them into a single data file that the main.py 
code can use <img src="https://github.com/Balistarama/04_SVHN-Image-Classifier/blob/master/images/tick.png?raw=true">
- Update data_processor.py to split out a few thousand examples for the validation dataset <img src="https://github.com/Balistarama/04_SVHN-Image-Classifier/blob/master/images/tick.png?raw=true">
- Configure main.py to properly import the pickled file <img src="https://github.com/Balistarama/04_SVHN-Image-Classifier/blob/master/images/tick.png?raw=true">
- Update data_processor.py to reshape the data so it's the right formate for the Neural Net <img src="https://github.com/Balistarama/04_SVHN-Image-Classifier/blob/master/images/tick.png?raw=true">
- Setup the Neural Net so that it accepts the new image tensors <img src="https://github.com/Balistarama/04_SVHN-Image-Classifier/blob/master/images/tick.png?raw=true">
- Setup TensorBoard
- Change final fully connected layer to 128 neurons
- Increase the models accuracy, current value: 16.13%
- Once model accuracy is high enough setup [TPOT](https://github.com/rhiever/tpot) and confirm that it's running at peak efficiency
- Deploy final solution to a Google Machine Learning Cloud

##INSTRUCTIONS:
 - Clone repository
 - Follow the data/README to download and munge the raw dataset
 - Run the src/main.py Neural Network

##REFERENCES:
You're welcome to use any and all code but a reference back to 
https://github.com/Balistarama would be appreciated
