# kaggle url : https://www.kaggle.com/uciml/pima-indians-diabetes-database/data
from cProfile import label
from turtle import pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read data from CSV
dataframe=pd.read_csv("diabetes.csv")

# แสดงข้อมูลมาดู 5 แถวแรก
#print(dataframe.head())
#print(dataframe.shape)

# read data from all attributes
x=dataframe.drop("Outcome",axis=1).values # ใส่ values เพื่อเปลี่ยนข้อมูลเป็น array 2 มิติ
#print(x) #แสดงค่าออกมาดู

# read outcome data (result)
y=dataframe['Outcome'].values
#print(y) #แสดงค่าออกมาดู

# split date into train data and test data 60-40
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4)

# ลองพิมพ์ออกมาดู
print(y_test.shape)

# creat and training model เมื่อเราไม่รู้ว่า knn รอบที่เท่าไหร่จะได้ค่าที่ดีที่สุด จึงสร้างค่า k_round มาเพื่อทดสอบ
k_round=np.arange(1,21) # จะได้ [1,2,3,...,6,7,8]

# สร้าง empty array มา 2 ตัวเพื่อเก็บค่าที่ได้ในแต่ละครั้งของแต่ละรอบลงใน array
train_score=np.empty(len(k_round))
test_score=np.empty(len(k_round))

for i,k in enumerate(k_round):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train,y_train)
    train_score[i]=knn.score(x_train,y_train) # เก็บผลลัพธ์ของแต่ละครั้งในการทดสอบ
    test_score[i]=knn.score(x_test,y_test) # เวลานำเอาไปใช้ เอาค่า k ของ test score ไปใช้
    print("K = ",i+1," Test Score = ",test_score[i]*100)

plt.title("Compare k Value in model")
plt.plot(k_round,test_score,label="Test Score")
plt.plot(k_round,train_score,label="Train Score")
plt.legend()
plt.xlabel("K Number")
plt.ylabel("Score")
plt.show()
