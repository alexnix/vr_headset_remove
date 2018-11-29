import numpy as np
import cv2
import time

face_cascade = cv2.CascadeClassifier('frontal_face.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        ss = img[y:y+h, x:x+w]
        ss = cv2.resize(ss, (28, 28))
        cv2.imwrite("frames_vr_on/" + str(time.time()) + ".png", ss)
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

    cv2.imshow('img',img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()