import cv2
import numpy as np

#img=cv2.imread("D:/python_/openCV/.idea/haarcascade/zayn.jpg")
img=cv2.imread("D:/python_/openCV/foto.jpg")


yuz_casc=cv2.CascadeClassifier('D:/python_/openCV/.idea/haarcascade/haarcascade_frontalface_default.xml') #yüz tanıma için kullanılack algoritma

# yüz tanıma için öğretilmiş olan algoritma xml dosyası

gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
yuzler=yuz_casc.detectMultiScale(gray_img,1.1, 4) # 4 kere kontrol etsin orda kaç tane yüz var

for(x,y,w,h) in yuzler:
    cv2.rectangle(img, (x,y),(x+w,y+h), (0,255,0),3)

cv2.imshow("yuzler", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
