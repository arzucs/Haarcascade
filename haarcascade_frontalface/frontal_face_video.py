import cv2
import numpy as np

cap=cv2.VideoCapture(0)
yuz_cascad=cv2.CascadeClassifier("D:/python_/openCV/.idea/haarcascade/haarcascade_frontalface_default.xml")

while True:
    ret, frame=cap.read()
    frame=cv2.flip(frame,1) # y eksenine göre simetrik alırız 
    griton=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    yuzler=yuz_cascad.detectMultiScale(griton,1.3,5)  #1.3= aldığımız görüntüyü yüzde kaç skalalsın. 3=videoda kaç kere yüz olduğunu kontrol edip göstersin.
    
    for (x,y,w,h) in yuzler:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,150,200),3) #w resmin genişliği, h resmin yüksekliği
    
    cv2.imshow("orjinal",frame)
    if cv2.waitKey(25) & 0xFF ==ord("q"):
        break

cap.release()
cv2.destroyAllWindows()    