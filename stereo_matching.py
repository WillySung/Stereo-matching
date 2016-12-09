import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
     gray_l = cv2.imread('aloeL.jpg',0)
     gray_r = cv2.imread('aloeR.jpg',0)
     gray_l = cv2.pyrDown(gray_l)  # downscale images for faster processing
     gray_r = cv2.pyrDown(gray_r)

     window_size = 3
     min_disp = 16
     num_disp = 112-min_disp
     stereo = cv2.StereoSGBM_create(
        minDisparity = min_disp,           # 視差の下限
        numDisparities = num_disp,        # 最大の上限
        blockSize = 16,
        P1 = 8*3*window_size**2,    # 視差のなめらかさを制御するパラメータ1
        P2 = 32*3*window_size**2,   # 視差のなめらかさを制御するパラメータ2
        disp12MaxDiff = 1,          # left-right 視差チェックにおけて許容される最大の差
        uniquenessRatio = 10,       # パーセント単位で表現されるマージン
        speckleWindowSize = 100,      # 視差領域の最大サイズ
        speckleRange = 32,          # それぞれの連結成分における最大視差値
     )

     #stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
     disp = stereo.compute(gray_l,gray_r).astype(np.float32) / 16.0    

     #plt.imshow(gray_l,'gray')
     #plt.imshow(disp,'disparity')
     #plt.show()
     cv2.imshow('gray_l',gray_l)
     cv2.imshow('gray_r',gray_r)
     cv2.imshow('disp', (disp-min_disp)/num_disp)
     cv2.waitKey(0)
     cv2.destroyAllWindows()
     
if __name__ == '__main__':
      main()   
