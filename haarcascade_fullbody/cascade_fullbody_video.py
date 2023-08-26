import cv2

body_casc=cv2.CascadeClassifier("D:/python_/openCV/.idea/haarcascade/haarcascade_fullbody/haarcascade_fullbody.xml")
kamera=cv2.VideoCapture("D:/python_/openCV/.idea/haarcascade/haarcascade_fullbody/video.mp4")


while True:
    ret, pencere=kamera.read()
    pencere=cv2.flip(pencere,1)
    griton=cv2.cvtColor(pencere, cv2.COLOR_BGR2GRAY)
    
    body=body_casc.detectMultiScale(griton, 1.05,5)
    
    for (x,y,w,h) in body:
        cv2.rectangle(pencere ,(x,y),(x+w,y+h),(0,255,255),3)
    
    cv2.imshow("ben", pencere)
    if cv2.waitKey(25) & 0xFF== ord("q"):
        break

pencere.release()
cv2.destroyAllWindows()