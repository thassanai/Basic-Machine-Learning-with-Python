import numpy as np
import matplotlib.pyplot as plt

rand_num = np.random
x = rand_num.rand(50)*10 # ทำการ random ตัวเลข 50 ตัว
c = rand_num.randn(50) # ทำการ random ตัวเลขติดลบ 50ตัว 

#print(x)
#print(c)

# ตั้งสมการ
y = 5*x+c

plt.plot(x,y) # การกระจายข้อมูลของ x,y
#plt.scatter(x,y) # การกระจายข้อมูลของ x,y
plt.xlabel('x') # ชื่อแกน x
plt.ylabel('y') # ชื่อแกน y
plt.title("Display graph y=5x+c ") # หัวข้อของกราฟ
plt.grid()
plt.show()