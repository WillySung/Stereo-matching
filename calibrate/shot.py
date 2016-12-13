import cv2
import time

AUTO = False  #set ture to auto snap shot
INTERVAL = 2  #snap shot time interval = 2 seconds

cv2.namedWindow("left")
cv2.namedWindow("right")
cv2.moveWindow("left",0,0)
cv2.moveWindow("right",800,0)

left_camera = cv2.VideoCapture(0)
right_camera = cv2.VideoCapture(1)

counter = 0
utc = time.time()
pattern = (8,6)   #the size of chessboard

def snap_shot(pos,frame):
    global counter
    filename = pos + "_" + str(counter) + ".jpg"

    cv2.imwrite(filename,frame)
    print(filename + " saved.")

while True:
    ret, left_frame = left_camera.read()
    ret, right_frame = right_camera.read()

    cv2.imshow("left", left_frame)
    cv2.imshow("right", right_frame)

    #auto snap shot mode
    now = time.time()
    if AUTO and now-utc >= INTERVAL:
         snap_shot("left", left_frame)
         snap_shot("right", right_frame)
         counter += 1
         utc = now
    
    #press s to snap shot
    key = cv2.waitKey(1)
    if key & 0xFF == ord("q"):
         break
    elif key & 0xFF == ord("s"):
         snap_shot("left", left_frame)
         snap_shot("right", right_frame)
         counter += 1

left_camera.release()
right_camera.release()
cv2.destroyAllWindows()
