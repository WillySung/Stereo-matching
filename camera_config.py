import cv2
import numpy as np

left_camera_matrix = np.array([[ 813.35575191,    0,          319.5871186 ],
                               [   0,          812.35926825,  243.55937314],
                               [   0,            0,            1        ]])

left_distortion = np.array([  2.07238902e-02,  -1.25611219e+00,   1.87849802e-03,  -4.28683753e-03,   7.11584927e+00])

right_camera_matrix = np.array([[ 817.28725852,    0,          317.534018  ],
                                [   0,          817.96578394,  251.73937654],
                                [   0,            0,            1        ]])

right_distortion = np.array([  1.42707517e-01,  -4.24991627e+00,  -1.37642002e-04,  -8.93958825e-03,   2.93141784e+01])

om = np.array([0.03944465, 0.20144, 1.52740131])
R = cv2.Rodrigues(om)[0]
T = np.array([5.62233136, -2.98129403, 25.81993809])
'''
R = np.array([[ 0.55203161, -0.05852514, 0.83176674],
             [-0.13085881,  0.97909187, 0.1557404 ],
             [-0.82349079, -0.19481763, 0.53283113]])
T = np.array([-20.71161089,  -3.72345748, 12.57491925])
'''
size = (640,480)

R1,R2,P1,P2,Q,roi1,roi2 = cv2.stereoRectify(left_camera_matrix,left_distortion,
                                            right_camera_matrix,right_distortion,
                                            size,R,T)

newcameramatrix1,roi1 = cv2.getOptimalNewCameraMatrix(left_camera_matrix,left_distortion,size,0,size)
newcameramatrix2,roi2 = cv2.getOptimalNewCameraMatrix(right_camera_matrix,right_distortion,size,0,size)

left_map1,left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix,left_distortion,None,newcameramatrix1,size,cv2.CV_16SC2)
right_map1,right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix,right_distortion,None,newcameramatrix2,size,cv2.CV_16SC2)

