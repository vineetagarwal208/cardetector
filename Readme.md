Required packages 
-Numpy
-SciPy
-Scikit-Image
-OpenCV 2

The package contains training and testing datasets, code for training your own car detector and using that on test images

1) Training
simply type 
python train.py
to train your object detector. After a successfull training attempt, you should see multiple .npy and .pkl files in the directory.

2) Testing on images of same size as training images (in the folder TestImages)

$ python slidingWindow.py <path to image>
for example 
$ python slidingWindow.py ./TestImages/test-6.png

3) Testing on images of different size than training set (in the folder TestImages_Scale)

$ python slidingWindow.py <path to image>
for example
$ python scaledSlidingWindow.py ./TestImages_Scaled/test-6.png





