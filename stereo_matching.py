# -*- coding: utf-8 -*-
import cv2
import numpy as np
import camera_config

cv2.namedWindow("left")
cv2.namedWindow("right")
cv2.namedWindow("depth")
cv2.moveWindow('left',0,0)
cv2.moveWindow('right',300,0)
cv2.moveWindow('depth',600,0)
cv2.createTrackbar('num', 'depth', 0, 10, lambda x:None)
cv2.createTrackbar('blockSize', 'depth', 5, 255, lambda x:None)

capL = cv2.VideoCapture(0)
capR = cv2.VideoCapture(1)
imgL = np.zeros((480,640,3),np.uint8)
imgR = np.zeros((480,640,3),np.uint8)

def callbackFunc(e,x,y,f,p):
    if e == cv2.EVENT_LBUTTONDOWN:
       print(threeD[y][x])
cv2.setMouseCallback("depth", callbackFunc, None)

while(True):
     capL.read(imgL)
     capR.read(imgR)

     imgL_rectified = cv2.remap(imgL, camera_config.left_map1, camera_config.left_map2, cv2.INTER_LINEAR)
     imgR_rectified = cv2.remap(imgR, camera_config.right_map1, camera_config.right_map2, cv2.INTER_LINEAR)

     gray_l = cv2.cvtColor(imgL_rectified, cv2.COLOR_BGR2GRAY)
     gray_r = cv2.cvtColor(imgR_rectified, cv2.COLOR_BGR2GRAY)
    
     gray_l = cv2.pyrDown(cv2.GaussianBlur(cv2.equalizeHist(gray_l),(5,5),0))
     gray_r = cv2.pyrDown(cv2.GaussianBlur(cv2.equalizeHist(gray_r),(5,5),0))

     # use two trackbar to adjust parameters
     num = cv2.getTrackbarPos("num", "depth")
     blockSize = cv2.getTrackbarPos("blockSize", "depth")
     if blockSize %2 == 0:
         blockSize += 1
     if blockSize < 5:
         blockSize = 5

     '''#for single image
     gray_l = cv2.imread('aloeL.jpg',0)
     gray_r = cv2.imread('aloeR.jpg',0)
     gray_l = cv2.pyrDown(gray_l)  # downscale images for faster processing
     gray_r = cv2.pyrDown(gray_r)
     
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

     stereo = cv2.StereoBM_create(numDisparities=16*num, blockSize=blockSize)
     disp = stereo.compute(gray_l,gray_r)  
     disp = cv2.normalize(disp,disp, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
     threeD = cv2.reprojectImageTo3D(disp.astype(np.float32)/16., camera_config.Q)     

     cv2.imshow('left',imgL_rectified)
     cv2.imshow('right',imgR_rectified)
     cv2.imshow('depth', disp)
     
     if cv2.waitKey(1) & 0xFF == ord('q'): break 

capL.release()
capR.release()
cv2.destroyAllWindows()
     
