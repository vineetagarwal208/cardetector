# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 12:10:13 2015

@author: agarwv1
"""


import cv2,os
from skimage.feature import hog
import sys
import sklearn.externals.joblib as joblib
clf=joblib.load('carSVM.pkl')

path=os.getcwd();
testimages=os.listdir(path+r'/TestImages')
img = sys.argv[1]
pixpcell=(8,8)
winsize=(40,100)
skip=6
found=[]

image = cv2.imread(sys.argv[1],0)
yiter=(image.shape[0]-winsize[0])/skip;
xiter=(image.shape[1]-winsize[1])/skip;

for i in range(0,xiter):
    for j in range(0,yiter):
        curwindow=image[j*skip:j*skip+winsize[0],i*skip:i*skip+winsize[1]]
        temp=hog(curwindow, orientations=9, pixels_per_cell=pixpcell, cells_per_block=(2, 2), visualise=False, normalise=True)
        if ((clf.predict_proba(temp)[0][1])>=0.90):
            print clf.predict_proba(temp)
            '''            
            cv2.imshow('img',curwindow)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            '''            
            found.append((i*skip,j*skip))
            

for s in found:
    cv2.rectangle(image,s,(s[0]+winsize[1],s[1]+winsize[0]),[0,255,0],2)

cv2.imshow('img',image)
cv2.waitKey(0)
cv2.destroyAllWindows()
