# pip3 install numpy
# pip3 install pillow
# pip3 install opencv-contrib-python
import numpy as np
from PIL import Image # แปลงภาพเป็น glayscale
import os # อ่านภาพจาก folder data
import cv2

def train_classifier(data_dir):
    path=[os.path.join (data_dir,f) for f in os.listdir(data_dir)] # ดึงไฟล์มาทั้งหมดใน folder
    faces=[] # สร้าง array เพื่อเก็บ faces
    ids=[] # สร้าง array เพื่อเก็บ id

    # แปลงภาพทั้งหมดเป็น grayscale
    for image in path:
        img=Image.open(image).convert("L")
        imageNp=np.array(img,'uint8') # assign ภาพลงไปใน array imageNp
        id=int(os.path.split(image)[1].split(".")[1])
        #print(str(id)) # แสดง id ออกมาทดสอบ
        faces.append(imageNp) # เก็บภาพที่ถูก train ไปแล้วใน faces
        ids.append(id)

    ids=np.array(ids)
    # algorithm ในการจดจำใบหน้า
    clf=cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces,ids)
    clf.write("classifier.xml")

train_classifier("data")
