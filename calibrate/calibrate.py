import numpy as np
import cv2
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*8,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpointsL = [] # 3d point in real world space
imgpointsL = [] # 2d points in image plane.
objpointsR = []
imgpointsR = []

images = glob.glob('left*.jpg')
for fname in images:
    imgL = cv2.imread(fname)
    grayL = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, cornersL = cv2.findChessboardCorners(grayL, (8,6),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpointsL.append(objp)

        corners2L = cv2.cornerSubPix(grayL,cornersL,(11,11),(-1,-1),criteria)
        imgpointsL.append(corners2L)

        # Draw and display the corners
        imgL = cv2.drawChessboardCorners(imgL, (8,6), corners2L,ret)
        cv2.imshow('imgL',imgL)

        #calibrate
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpointsL, imgpointsL, grayL.shape[::-1], None, None)
        print("camera matrix:\n", mtx)
        print("distortion coefficients:", dist.ravel())
        print(rvecs)
        print(tvecs)

        cv2.waitKey(500)

images = glob.glob('right*.jpg')
for fname in images:
    imgR = cv2.imread(fname)
    grayR = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, cornersR = cv2.findChessboardCorners(grayR, (8,6),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpointsR.append(objp)

        corners2R = cv2.cornerSubPix(grayR,cornersR,(11,11),(-1,-1),criteria)
        imgpointsR.append(corners2R)

        # Draw and display the corners
        imgR = cv2.drawChessboardCorners(imgR, (8,6), corners2R,ret)
        cv2.imshow('imgR',imgR)

        #calibrate
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpointsR, imgpointsR, grayR.shape[::-1], None, None)
        print("camera matrix:\n", mtx)
        print("distortion coefficients:", dist.ravel())
        print(rvecs)
        print(tvecs)

        cv2.waitKey(500)

cv2.destroyAllWindows()
