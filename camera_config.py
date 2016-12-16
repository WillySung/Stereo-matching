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

left_rotation_vector = np.array([ 0.03944465,  0.20144,    1.52740131])
right_rotation_vector = np.array([ 0.43235342,  0.09365809, -0.41773828])
left_translation_vector = np.array([  5.62233136,  -2.98129403,  25.81993809])
right_translation_vector = np.array([ -3.29689611,  -1.03348628,  21.56224497])

rvec1 = cv2.Rodrigues(left_rotation_vector)[0]
rvec2 = cv2.Rodrigues(right_rotation_vector)[0]
R = cv2.Rodrigues(rvec2*rvec1)[0]
T = np.float32(rvec2*left_translation_vector + right_translation_vector)

size = (640,480)

R1,R2,P1,P2,Q,roi1,roi2 = cv2.stereoRectify(left_camera_matrix,left_distortion,
                                            right_camera_matrix,right_distortion,
                                            size,R,T,alpha=0)

left_map1,left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix,left_distortion,R1,P1,size,cv2.CV_16SC2)
right_map1,right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix,right_distortion,R2,P2,size,cv2.CV_16SC2)

