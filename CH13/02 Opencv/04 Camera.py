import cv2

cap = cv2.VideoCapture(2) # สั่งเปิดตัวกล้อง ระบุ index ของตัวกล้อง
#cap = cv2.VideoCapture('googlecar.mp4') # เปิด file

# ส่งภาพแต่ละ frame จากกล้องมาเรื่อยๆ
while(True):
    ret,frame = cap.read()
    # ต้องการเปลี่ยนวีดีโอเป็น gray scale
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('Camera',frame)
    if(cv2.waitKey(1) &0xFF== ord('q')): # ให้ออกจาก stream เมื่อกดปุ่ม q
        break

cap.release()
cv2.destroyAllWindows()
