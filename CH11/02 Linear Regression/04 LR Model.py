import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

rand_num = np.random

# step 1 การจำลองข้อมูล
x = rand_num.rand(50)*10 # ทำการ random ตัวเลข 50 ตัว
c = rand_num.randn(50) # ทำการ random ตัวเลขติดลบ 50ตัว 
print(x)
print(c)

# y = 5*x+c

# # step 2 สร้าง linear regression model
# model = LinearRegression()

# #print(x)
# x_new = x.reshape(-1,1)
# #print(x_new)

# # step 3 train model
# model.fit(x_new,y) # train algorithm
# print(model.score(x_new,y)) # แสดงค่า R square เพื่อเป็นข้อมูลในการตัดสินใจวัดความแม่นยำ ค่าตัวแปรตอบสนอง (y) 0-100

# # step 4 test model
# x_test = np.linspace(-1,11) # หาข้อมูลตัวอย่างมาทดสอบ
# x_test_new = x_test.reshape(-1,1) # เปลี่ยนข้อมูลให้เป็น array 2 มิติ

# y_test= model.predict(x_test_new) # test model ใช้คำสั่ง predict

# # step 5 analysis model and result
# plt.scatter(x,y)
# plt.plot(x_test,y_test,'-r')
# plt.show()

