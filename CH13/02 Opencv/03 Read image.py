import cv2

img = cv2.imread("tiger.jpg",cv2.IMREAD_GRAYSCALE) # GRAY_SCALE
cv2.imshow('Tiger Picture',img)

cv2.waitKey(0) # รอกดปุ่มอะไรสักอย่าง
cv2.destroyAllWindows # ปิดหน้าต่างที่เปิดไว้

# เขียนรูปภาพลงไฟล์
#cv2.imwrite('saveimage.png',img)
