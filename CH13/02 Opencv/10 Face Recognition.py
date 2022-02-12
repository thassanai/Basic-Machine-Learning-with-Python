### copy มาจากไฟล์ FR [create dataset].py
import cv2

# load the required XML classifiers
# https://github.com/opencv/opencv/tree/master/data/haarcascades
faceCascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

## function เพื่อทำการเก็บใบหน้าจาก function face_detect เอามา training
def generate_dataset(img,id,img_id):
    cv2.imwrite("data/pic"+str(id)+"."+str(img_id)+".jpg",img) ## เป็น id ของใบหน้าที่เจอ

# function เพื่อทำการวาดขอบเขตของใบหน้าเมื่อเจอใบหน้าของมนุษย์
def draw_face_boundary(img,classifier,scaleFactor,minNeighbors,color,clf): ### ลบ text ออก
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # อ่านภาพหรือวีดีโอแล้วแปลงให้เป็น glayscale
    features=classifier.detectMultiScale(gray,scaleFactor,minNeighbors)
    coords=[] #สร้าง array เพื่อเก็บตำแหน่งแกน x,y
    for (x,y,w,h) in features:
        # วาดสี่เหลี่ยมผืนผ้าบนใบหน้าที่พบ
        cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
        ### Prediction
        id,v_con=clf.predict(gray[y:y+h,x:x+w]) ### ทำการ prediction ภาพกับวีดีโอหรือกล้องที่อ่านมา v_con คือ ค่า confidence หรือค่าความแม่นยำ
        #### if id==1: #### ซึ่งจะไม่มีความแม่นยำ เพราะตรวจสอบแค่ id
        #### แสดงค่าความแม่นยำ (confidence)
        #v_con = "  {0}%".format(round(100-v_con))

        ####    cv2.putText(img,"Steve Jobs",(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,1)
        if v_con <= 60:
            cv2.putText(img,"Pui:Thassanai",(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,1)
        else :
            cv2.putText(img,"Unknow",(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,1)
        
        # แสดงค่า confidence
        print(str(v_con))
        coords=[x,y,w,h] ## ถ้า detect ใบหน้าครบทุกส่วนจะมีจุด coords ครบทุกค่า
    return img,coords

# มีหน้าที่ในการเรียกใช้ draw_face_boundary และเรียกใช้ function generate_dataset
def face_detect(img,faceCascade,img_id,clf): ### รับค่า clf จาก traindata
    img,coords=draw_face_boundary(img,faceCascade,1.1,10,(255,0,0),clf) ### ส่งค่า clf ไปเพื่อเปรียบกับหน้าจริงกับ dataset
    ## ตรวจสอบว่าเจอใบหน้าหรือเปล่า จะดูว่ามีความยาวใน coords ครบ 4 ค่าหรือเปล่า ถ้าครบแสดงว่าตรวจจับเจอใบหน้า
    if len(coords)==4:
        id=1
        result = img[coords[1]:coords[1]+coords[3],coords[0]:coords[0]+coords[2]] ## img[y:y+h,x:x+w]
        ### generate_dataset(result,id,img_id) ## เรียกใช้ function generate_dataset เปลี่ยน img-> result
    return img

img_id=0 ## ประกาศเป็นตัวแปร global variable assign ค่าให้เท่ากับ 0
cap=cv2.VideoCapture(0)

### เพิ่ม
clf=cv2.face.LBPHFaceRecognizer_create()
clf.read("classifier.xml") # อ่านไฟล์ xml ที่เรา train data ไว้แล้วมาใช้งาน

while (True):
    ret,frame=cap.read()
    ### โยน clf เข้าไปทำงาน
    frame=face_detect(frame,faceCascade,img_id,clf) ## ส่งแต่ละ frame ไปยัง function face_detect ต่อมาเพิ่ม img_id
    cv2.imshow('Realtime Face Detection',frame)
    img_id+=1 ## เพื่อไม่ให้ภาพบันทึกซ้ำ นับ img_id เพิ่มเรื่อย ๆ
    if(cv2.waitKey(1) & 0xFF==ord('q')):
        break

cap.release()
cv2.destroyAllWindows()
