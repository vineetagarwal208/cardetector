# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 13:18:21 2015

@author: agarwv1
"""


import cv2,os,sys
import numpy as np

from skimage.feature import hog
import sklearn.externals.joblib as joblib

clf=joblib.load('carSVM.pkl')
pixpcell=(8,8)

resizer=1.3
winsize=(40,100)
image = cv2.imread(sys.argv[1],0)
imgsize=image.shape

winsizelist=[]
found=[]
probabilities=[]

while (winsize[0]*resizer<imgsize[0])and(winsize[1]*resizer<imgsize[1]):
    winsize=(int(winsize[0]*resizer),int(winsize[1]*resizer))
    winsizelist.append(winsize)
    
for winsize in winsizelist:
    skip=winsize[0]/8
    yiter=(image.shape[0]-winsize[0])/skip;
    xiter=(image.shape[1]-winsize[1])/skip;
    
    for i in range(0,xiter):
        for j in range(0,yiter):
            curwindow=image[j*skip:j*skip+winsize[0],i*skip:i*skip+winsize[1]]
            curwindow=cv2.resize(curwindow,(100,40))
#            cv2.imshow('img',curwindow)
#            cv2.waitKey(0)
#            cv2.destroyAllWindows()
            temp=hog(curwindow, orientations=9, pixels_per_cell=pixpcell, cells_per_block=(2, 2), visualise=False, normalise=True)
            prob=clf.predict_proba(temp)[0][0]            
            if (prob<=1e-1):
                #cv2.rectangle(image,(i*skip,j*skip),(i*skip+winsize[1],j*skip+winsize[0]),[0,255,0],3)
                print prob
                found.append((i*skip,j*skip,winsize[0],winsize[1]))
                probabilities.append(prob)
                

rect=found[np.argmin(probabilities)]
cv2.rectangle(image,(rect[0],rect[1]),(rect[0]+rect[3],rect[1]+rect[2]),[0,255,0],3)

cv2.imshow('img',image)
cv2.waitKey(0)
cv2.destroyAllWindows()
