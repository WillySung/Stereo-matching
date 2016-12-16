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
        #cv2.imshow('imgL',imgL)

        #calibrate
        #ret, mtxL, distL, rvecs, tvecs = cv2.calibrateCamera(objpointsL, imgpointsL, grayL.shape[::-1], None, None)
        #print("camera matrix:\n", mtx)
        #print("distortion coefficients:", dist.ravel())
        #print(rvecs)
        #print(tvecs)

        #cv2.waitKey(500)

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
        #cv2.imshow('imgR',imgR)

        #calibrate
        #ret, mtxR, distR, rvecs, tvecs = cv2.calibrateCamera(objpointsR, imgpointsR, grayR.shape[::-1], None, None)
        #print("camera matrix:\n", mtx)
        #print("distortion coefficients:", dist.ravel())
        #print(rvecs)
        #print(tvecs)

        #cv2.waitKey(500)

cameraMatrix1=np.array([[ 813.35575191,    0,          319.5871186 ],
                        [   0,          812.35926825,  243.55937314],
                        [   0,            0,            1        ]])
distCoeffs1=np.array([  2.07238902e-02,  -1.25611219e+00,   1.87849802e-03,  -4.28683753e-03,   7.11584927e+00])
cameraMatrix2=np.array([[ 817.28725852,    0,          317.534018  ],
                        [   0,          817.96578394,  251.73937654],
                        [   0,            0,            1        ]])
distCoeffs2=np.array([  1.42707517e-01,  -4.24991627e+00,  -1.37642002e-04,  -8.93958825e-03,   2.93141784e+01])
flags = (cv2.CALIB_FIX_ASPECT_RATIO + cv2.CALIB_ZERO_TANGENT_DIST + cv2.CALIB_SAME_FOCAL_LENGTH)
retval,cameraMatrix1,distCoeffs1,cameraMatrix2,distCoeffs2,R,T,E,F = cv2.stereoCalibrate(objpointsL,imgpointsL, imgpointsR,cameraMatrix1,distCoeffs1,cameraMatrix2,distCoeffs2, (640,480))
print(R)
print(T)
cv2.destroyAllWindows()
