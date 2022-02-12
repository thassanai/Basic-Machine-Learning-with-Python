import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5,5,100) # สร้างกลุ่มตัวเลขจาก -5 ถึง 5 array จำนวน 100 ชุด
# ลองเปลี่ยนให้ข้อมูลมีการกระจาย
# print(x)

# สร้างสมการ y = ax + b 
y = 5*x + 4

plt.scatter(x,y) # การกระจายข้อมูลของ x,y
plt.xlabel('x') # ชื่อแกน x
plt.ylabel('y') # ชื่อแกน y
plt.legend(loc="upper left") # label แสดงในส่วนบนซ้าย
plt.title("Display graph y=5x+4 ") # หัวข้อของกราฟ
plt.grid()
plt.show()