import cv2

cars_casc=cv2.CascadeClassifier("D:/python_/openCV/.idea/haarcascade/haarcascade_car/cars.xml")

cap=cv2.VideoCapture("D:/python_/openCV/.idea/haarcascade/haarcascade_car/carvideo.mp4")

while True:
    ret,frame= cap.read()
    griton=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cars=cars_casc.detectMultiScale(griton,1.2,4) #1.2= aldığımız görüntüyü yüzde kaç skalalsın. 4=videoda kaç kere yüz olduğunu kontrol edip göstersin. bu değerler değiştirilerek doğurluk oranı ayarlanabilir.
                                                 
    for (x,y,w,h) in cars:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)
        
    cv2.imshow("araba",frame)
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break 
    
cap.release()
cv2.destroyAllWindows()       