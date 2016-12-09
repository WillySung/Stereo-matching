import cv2
import sys

cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier(cascPath)

cam = cv2.VideoCapture(0)

while(True):
     ret_val, img = cam.read()

     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

     faces = faceCascade.detectMultiScale(
          gray,
          scaleFactor=1.5,
          minNeighbors=5,
          minSize=(30,30),
          flags=cv2.CASCADE_SCALE_IMAGE
     )

     for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,0),2)
         
     cv2.imshow('my webcam', img)
     if cv2.waitKey(1) & 0xFF == ord('q'):
            break # press q to quit
cam.release()
cv2.destroyAllWindows()

