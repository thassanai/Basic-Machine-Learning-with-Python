# kaggle url : https://www.kaggle.com/uciml/pima-indians-diabetes-database/data
from cProfile import label
from turtle import pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read data from CSV
dataframe=pd.read_csv("/Users/thassanai/Desktop/Introduction_to_Python/Data Set/diabetes.csv")

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
#print(y_test.shape)

knn=KNeighborsClassifier(n_neighbors=8)
knn.fit(x_train,y_train)

# prediction
y_predict=knn.predict(x_test)

#print(classification_report(y_test,y_predict))
#print(confusion_matrix(y_test,y_predict))
print(pd.crosstab(y_test,y_predict,rownames=['Actually'],margins=True))
