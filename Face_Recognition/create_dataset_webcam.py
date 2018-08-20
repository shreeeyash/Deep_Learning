import cv2
import numpy

face_cascade = cv2.CascadeClassifier('/home/shreeyash/opencv-3.4.1/data/haarcascades/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
i=0
while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    if len(faces)>0:
        i=i+1;
        cv2.imwrite('/home/shreeyash/PycharmProjects/ML/Face Recognition/dataset/Shreeyash/syg{}.jpg'.format(i),img)

    cv2.imshow('frames', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()