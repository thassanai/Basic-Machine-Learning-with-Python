import cv2

# https://docs.opencv.org/4.x/dc/da5/tutorial_py_drawing_functions.html
img=cv2.imread('tiger.jpg')

#img=cv2.line(img,(0,0),(800,800),(0,0,255),20) # วาดเส้นตรง
#img=cv2.arrowedLine(img,(0,0),(800,800),(255,0,0),20) # วาดลูกศร
img=cv2.rectangle(img,(384,0),(510,128),(0,0,255),5) # วาดสี่เหลี่ยม สีแดง หนา 5 px
#img=cv2.circle(img,(564,989),100,(0,255,0),10) # วาดวงกลม
#img=cv2.putText(img,"OpenCv",(10,100),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2) # วาดข้อความลงไป

cv2.imshow("Tiger",img)

cv2.waitKey(0)
cv2.destroyAllWindows()
