# -*- coding: utf-8 -*-
import cv2
import numpy as np

capL = cv2.VideoCapture(0)
capR = cv2.VideoCapture(1)
imgL = np.zeros((480,640,3),np.uint8)
imgR = np.zeros((480,640,3),np.uint8)

while(True):
     capL.read(imgL)
     capR.read(imgR)
     gray_l = cv2.pyrDown(cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY))
     gray_r = cv2.pyrDown(cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY))

     '''#for single image
     gray_l = cv2.imread('aloeL.jpg',0)
     gray_r = cv2.imread('aloeR.jpg',0)
     gray_l = cv2.pyrDown(gray_l)  # downscale images for faster processing
     gray_r = cv2.pyrDown(gray_r)
     '''
     '''
     #StereoSGBM setting
     window_size = 5
     min_disp = 16
     num_disp = 112-min_disp
     stereo = cv2.StereoSGBM_create(
        minDisparity = min_disp,    
        numDisparities = num_disp,  
        blockSize = 16,
        P1 = 8*3*window_size**2,   
        P2 = 32*3*window_size**2,  
        disp12MaxDiff = 1,         
        uniquenessRatio = 10,      
        speckleWindowSize = 100,   
        speckleRange = 32,         
     )'''

     stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
     disp = stereo.compute(gray_l,gray_r)  

     #disp = stereo.compute(gray_l,gray_r).astype(np.float32) / 16.0    

     cv2.imshow('gray_l',gray_l)
     cv2.imshow('gray_r',gray_r)
     cv2.imshow('disp', disp/255)
     
     if cv2.waitKey(1) & 0xFF == ord('q'): break 

capL.release()
capR.release()
cv2.destroyAllWindows()
     
