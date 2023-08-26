import cv2

yuz_casc=cv2.CascadeClassifier("D:/python_/openCV/.idea/haarcascade/haarcascade_eyeandface/haarcascade_frontalface_default.xml")
gozz_casc=cv2.CascadeClassifier("D:/python_/openCV/.idea/haarcascade/haarcascade_eyeandface/haarcascade_eye.xml")

cap=cv2.VideoCapture(0)

while True:
    ret,frame=cap.read()
    frame=cv2.flip(frame,1)
    griton=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    yuzler=yuz_casc.detectMultiScale(griton,1.3,5)
    for (x,y,w,h) in yuzler:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(50,150,255),3)
                                   #dikdörtgenin sol üst köşe sağ alt köşesi
        roi_griton=griton[y:y+h,x:x+w]
        roi_renkli=frame[y:y+h,x:x+w]
        gozlar=gozz_casc.detectMultiScale(roi_griton,1.3,3)
        for (x1,y1,w1,h1) in gozlar:
            cv2.rectangle(roi_renkli, (x1,y1),(x1+w1,y1+h1),(150,200,0),2)
    cv2.imshow("kare",frame)
    if cv2.waitKey(25) & 0xFF ==ord("q"):
        break
cap.release()
cv2.destroyAllWindows()