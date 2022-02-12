import cv2

# เรียกใช้ license plate cascade
lpcascade= cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")
min_area=1000

while True:
    img=cv2.imread("lp06.jpg")
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    licensePlate= lpcascade.detectMultiScale(img,1.1,5)

    for (x,y,w,h) in licensePlate:
        area=w*h
        if area > min_area:
            cv2.rectangle(img,(x-50,y), (x+w,y+h), (0,0,255),1)
            cv2.putText(img, "License Plate", (x,y-35),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)
            img_crop=img[y:y+h,x:x+w-50]
            cv2.imshow("License Plate",img_crop)
            cv2.imshow("LP Original",img)


