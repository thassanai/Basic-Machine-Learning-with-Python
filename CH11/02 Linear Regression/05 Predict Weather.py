import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.io.formats import style
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

weather_dataset=pd.read_csv("Weather.csv")
#print(weather_dataset.shape) # ลอง print ขนาดของ dataset มาดู

#weather_dataset.plot(x="MinTemp",y="MaxTemp",style="o") # ใส่ style="o" เพื่อ plot เป็นจุด โดยต้อง pip3 install Jinja2
#plt.title("Min & Max Temp")
#plt.xlabel("Min Temp")
#plt.ylabel("Max Temp")
#plt.show()

# เราสามารถดูค่าทางสถิติได้ โดยใช้ pandas
#print(weather_dataset.describe())

# train & test set
x = weather_dataset["MinTemp"].values.reshape(-1,1) # เปลี่ยนข้อมูลให้เป็น array 2 มิติ
y = weather_dataset["MaxTemp"].values.reshape(-1,1)

# 80% - 20% # test_size คือจำนวน test set ที่ต้องการทดสอบ 20%
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

# training model
model = LinearRegression()
model.fit(x_train,y_train)

# test model ใช้คำสั่ง predict
y_predict = model.predict(x_test) # ตัวแปรที่เราทราบค่าคือ x_test

# ดูการกระจายข้อมูล
# plt.scatter(x_test,y_test)
# plt.plot(x_test,y_predict,color="red",linewidth="2")
# plt.show()

# compare true data with predict data สร้าง dataframe ใหม่ เพื่อเปรียบเทียบ
# นำเอา .flatten เพื่อแปลงข้อมูลจาก 2 มิติ เป็น 1 มิติ
df = pd.DataFrame({'Actually':y_test.flatten(),'Predicted':y_predict.flatten()})
#print(df.head()) # แสดงออกมา 5 แถวแรก

df1 = df.head(20)
df1.plot(kind="bar",figsize=(16,10))
plt.show()
