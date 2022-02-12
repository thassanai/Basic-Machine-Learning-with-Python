# pip3 install scipy
from numpy import number
from numpy.lib.npyio import load
from scipy.io import loadmat
import matplotlib.pyplot as plt

mnist_raw = loadmat("/Users/thassanai/Desktop/Introduction_to_Python/Data Set/mnist-original.mat") # เก็บข้อมูลไว้ที่ raw file

print(mnist_raw) # ลองแสดงข้อมูลมาดู ข้อมูลสำหรับการ traning data 1-70000, test data 70000-80000 ไว้สำหรับทดสอบ

# อยากได้แค่ data ไม่ได้อยากได้ header ทำอย่างไร
mnist ={
    "data": mnist_raw["data"].T, # matrix transpos สลับ row -> column
    "target": mnist_raw["label"][0] #เอาเฉพาะ label
}

print(mnist["data"].shape)

x=mnist["data"]
y=mnist["target"]
# หรือเขียนเป็นแบบนี้ได้เลย
# x,y = mnist["data"],mnist["target"]

number=x[15000] # เลข 5 จะอยู่ที่ตำแหน่ง 35000 เลข 0 อยู่ที่ 2000 เลข 1 อยู่ที่ 10000
number_image=number.reshape(28,28)  # แปลงขนาดให้เป็น array 2 มิติ ขนาด 28x28 pixels

print(y[15000]) # อยากดูว่าแสดงผลเป็นรูปภาพตัวเลขอะไร

plt.imshow(
    number_image,
    cmap=plt.cm.binary,
    interpolation="nearest"
)
plt.show()
