# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 09:16:22 2015

@author: agarwv1
"""
import os
from skimage.feature import hog
import numpy as np
import cv2
import pickle

#parameter for HOG features
pixpcell=(8,8)
path=os.getcwd();
#supported formats
formats = ['PNG','png','JPG','jpg','JPEG','jpeg']
# list all images in pos and neg folders
positives=os.listdir(path+r'/pos')
negatives=os.listdir(path+r'/neg')
positives = [x for x in positives if x.split('.')[-1] in formats]
negatives = [x for x in negatives if x.split('.')[-1] in formats]

print('Visualizing sample HOG features')
image=cv2.imread(path+r'/pos/'+positives[5],0)
sample,viz=hog(image, orientations=9, pixels_per_cell=pixpcell, cells_per_block=(2, 2), visualise=True, normalise=True)
viz=cv2.resize(viz,(300,120))
image=cv2.resize(image,(300,120))
cv2.imshow('image',image)
cv2.imshow('features',viz)
cv2.waitKey(0)
cv2.destroyAllWindows()
#Declare space for storing features
features=np.zeros((len(positives)+len(negatives),sample.size))
labels=np.zeros((len(positives)+len(negatives)))
i=0

print('Generating features for positive samples')
for s in positives:
    if(s[-3:]=='png'):
        image=cv2.imread(path+r'/pos/'+s,0)        
        features[i]=hog(image, orientations=9, pixels_per_cell=pixpcell, cells_per_block=(2, 2), visualise=False, normalise=True)
        labels[i]=1
        i=i+1
        

print('Generating features for negative samples')
for s in negatives:
    if(s[-3:]=='png'):
        image=cv2.imread(path+r'/neg/'+s,0)
        features[i]=hog(image, orientations=9, pixels_per_cell=pixpcell, cells_per_block=(2, 2), visualise=False, normalise=True)
        labels[i]=-1
        i=i+1

from sklearn.svm import SVC
clf=SVC(C=1.0, kernel='linear',probability=True)
print('Training SVM')
clf.fit(features,labels)

import sklearn.externals.joblib as joblib
print('saving SVM model')
joblib.dump(clf, 'carSVM.pkl')

