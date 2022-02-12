import cv2

# load the required XML classifiers
# https://github.com/opencv/opencv/tree/master/data/haarcascades
faceCascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

## function เพื่อทำการเก็บใบหน้าจาก function face_detect เอามา training
def generate_dataset(img,id,img_id):
    cv2.imwrite("data/pic."+str(id)+"."+str(img_id)+".jpg",img) ## เป็น id ของใบหน้าที่เจอ

# function เพื่อทำการวาดขอบเขตของใบหน้าเมื่อเจอใบหน้าของมนุษย์
def draw_face_boundary(img,classifier,scaleFactor,minNeighbors,color,text):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # อ่านภาพหรือวีดีโอให้แปลงเป็น glayscale
    features=classifier.detectMultiScale(gray,scaleFactor,minNeighbors)
    coords=[] #สร้าง array เพื่อเก็บตำแหน่งแกน x,y
    for (x,y,w,h) in features:
        # วาดสี่เหลี่ยมผืนผ้าบนใบหน้าที่พบ
        cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
        cv2.putText(img,text,(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,1)
        coords=[x,y,w,h] ## ถ้า detect ใบหน้าครบทุกส่วนจะมีจุด coords ครบทุกค่า
    return img,coords

# มีหน้าที่ในการเรียกใช้ draw_face_boundary และเรียกใช้ function generate_dataset
def face_detect(img,faceCascade,img_id):
    img,coords=draw_face_boundary(img,faceCascade,1.1,10,(255,0,0),"Face")
    ## ตรวจสอบว่าเจอใบหน้าหรือเปล่า จะดูว่ามีความยาวใน coords ครบ 4 ค่าหรือเปล่า ถ้าครบแสดงว่าตรวจจับเจอใบหน้า
    if len(coords)==4:
        id=1
        result = img[coords[1]:coords[1]+coords[3],coords[0]:coords[0]+coords[2]] ## img[y:y+h,x:x+w]
        generate_dataset(result,id,img_id) ## เรียกใช้ function generate_dataset เปลี่ยน img-> result
    return img

img_id=0 ## ประกาศเป็นตัวแปร global variable assign ค่าให้เท่ากับ 0
cap=cv2.VideoCapture(2)
while (True):
    ret,frame=cap.read()
    frame=face_detect(frame,faceCascade,img_id) ## ส่งแต่ละ frame ไปยัง function face_detect ต่อมาเพิ่ม img_id
    cv2.imshow('Realtime Face Detection',frame)
    img_id+=1 ## เพื่อไม่ให้ภาพบันทึกซ้ำ นับ img_id เพิ่มเรื่อย ๆ
    if(cv2.waitKey(1) & 0xFF==ord('q')):
        break

cap.release()
cv2.destroyAllWindows()
