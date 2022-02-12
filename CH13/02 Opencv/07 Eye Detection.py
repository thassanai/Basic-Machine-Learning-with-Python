import cv2

# load the required XML classifiers
# https://github.com/opencv/opencv/tree/master/data/haarcascades
faceCascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
#eyeCascade=cv2.CascadeClassifier("haarcascade_eye.xml") # โหลด eye classifier
eyeCascade=cv2.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml") # โหลด eye classifier

# function เพื่อทำการวาดขอบเขตของใบหน้าเมื่อเจอใบหน้าของมนุษย์
def draw_face_boundary(img,classifier,scaleFactor,minNeighbors,color,text):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # อ่านภาพหรือวีดีโอให้แปลงเป็น glayscale
    features=classifier.detectMultiScale(gray,scaleFactor,minNeighbors)
    coords=[] #สร้าง array เพื่อเก็บตำแหน่งแกน x,y
    for (x,y,w,h) in features:
        # วาดสี่เหลี่ยมผืนผ้าบนใบหน้าที่พบ
        cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
        cv2.putText(img,text,(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,1)
        coords=[x,y,w,h]
    return img,coords

# มีหน้าที่ในการเรียกใช้ draw_face_boundary
def face_detect(img,faceCascade):
    img,coords=draw_face_boundary(img,faceCascade,1.1,10,(255,0,0),"Face")
    img,coords=draw_face_boundary(img,eyeCascade,1.1,10,(0,0,255),"Eye") # eyeCascade
    return img

cap=cv2.VideoCapture(2)
while (True):
    ret,frame=cap.read()
    frame=face_detect(frame,faceCascade) # ส่งแต่ละ frame ไปยัง function
    cv2.imshow('Realtime Face Detection',frame)
    if(cv2.waitKey(1) & 0xFF==ord('q')):
        break

cap.release()
cv2.destroyAllWindows()
