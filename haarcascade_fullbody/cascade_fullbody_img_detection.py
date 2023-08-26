import cv2

body_casc=cv2.CascadeClassifier("D:/python_/openCV/.idea/haarcascade/haarcascade_fullbody/haarcascade_fullbody.xml")

img=cv2.imread("D:/python_/openCV/.idea/haarcascade/haarcascade_fullbody/full.jpg")

griton=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
body=body_casc.detectMultiScale(griton, 1.05,3)

for (x,y,w,h) in body:
    cv2.rectangle(img, (x,y),(x+w,y+h),(50,0,200),3)
    
cv2.imshow("full body", img)
cv2.waitKey(0)
cv2.destroyAllWindows()