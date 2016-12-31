The Street View House Numbers (SVHN) Dataset
--------------------------------------------------------------------------------
Downloaded from http://ufldl.stanford.edu/housenumbers/ on 29th November 2016.

SVHN is a real-world image dataset for developing machine learning and object 
recognition algorithms with minimal requirement on data preprocessing and 
formatting. It can be seen as similar in flavor to MNIST (e.g., the images 
are of small cropped digits), but incorporates an order of magnitude more 
labeled data (over 600,000 digit images) and comes from a significantly harder, 
unsolved, real world problem (recognizing digits and numbers in natural 
scene images). SVHN is obtained from house numbers in Google Street View images. 

10 classes, 1 for each digit. Digit '1' has label 1, '9' has label 9 and '0' has label 10.
73257 digits for training, 26032 digits for testing, and 531131 additional, somewhat 
less difficult samples, to use as extra training data
Format 2: Cropped Digits: train_32x32.mat, test_32x32.mat , extra_32x32.mat (Note: for non-commercial use only)

DIRECT LINKS:
--------------------------------------------------------------------------------
- http://ufldl.stanford.edu/housenumbers/train_32x32.mat
- http://ufldl.stanford.edu/housenumbers/test_32x32.mat
- hhttp://ufldl.stanford.edu/housenumbers/extra_32x32.mat

AFTER PROCESSING:
--------------------------------------------------------------------------------
- "data_processor.py" was then run to munge data into SVHN.pickle

REFERENCES:
--------------------------------------------------------------------------------
Please cite the following reference in papers using this dataset:
Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Bo Wu, Andrew Y. Ng 
Reading Digits in Natural Images with Unsupervised Feature Learning NIPS 
Workshop on Deep Learning and Unsupervised Feature Learning 2011. 
(http://ufldl.stanford.edu/housenumbers/nips2011_housenumbers.pdf)

Please use http://ufldl.stanford.edu/housenumbers as the URL for this site 
when necessary. For questions regarding the dataset, please contact 
streetviewhousenumbers@gmail.com
