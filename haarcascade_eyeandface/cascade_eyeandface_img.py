import cv2
import numpy as np

img=cv2.imread("D:/python_/openCV/.idea/haarcascade/img.jpg")
goz_casc=cv2.CascadeClassifier("D:/python_/openCV/.idea/haarcascade/haarcascade_eye.xml")
yuz_casc=cv2.CascadeClassifier('D:/python_/openCV/.idea/haarcascade/haarcascade_frontalface_default.xml')

griton=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gozler=goz_casc.detectMultiScale(griton,1.1,3)
yuzler=yuz_casc.detectMultiScale(griton,1.3,3)
for (x,y,w,h) in gozler:
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,150,250),3)

for (x,y,w,h) in yuzler:
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,150,250),3)

cv2.imshow("zayn", img)
cv2.waitKey(0)
cv2.destroyAllWindows()