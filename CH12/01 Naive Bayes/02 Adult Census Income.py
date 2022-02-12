#https://www.kaggle.com/uciml/adult-census-income
from dataclasses import dataclass
import pandas as pd
# เปลี่ยน label จากตัวหนังสือให้เป็นตัวเลข โดยใช้ LabelEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

def cleandata(dataset):
    for column in dataset.columns:
        if dataset[column].dtype == type(object):
            le = LabelEncoder()
            dataset[column]=le.fit_transform(dataset[column])
    return dataset

# สร้าง function เปลี่ยน age - native.country เป็น attribute ส่วน income เป็น class
def split_feature_class(dataset,feature):
    features=dataset.drop(feature,axis=1) # เอา Income ออกไป , เอาเฉพาะ column ถ้า axis = 0 เอาเฉพาะแถว
    lebels=dataset[feature].copy() # เอาเฉพาะ Income อย่างเดียว
    return features,lebels


income_dataset=pd.read_csv("adult.csv")
# ทดสอบแสดง dataset มาดูว่าได้หรือไม่
#print(income_dataset.head())

# เปลี่ยน label จากตัวหนังสือให้เป็นตัวเลข โดยใช้ LabelEncoder
income_dataset=cleandata(income_dataset)
#print(income_dataset.head())

# แบ่ง dataset ออกเป็น 2 กลุ่ม train,test
train_set,test_set = train_test_split(income_dataset,test_size=0.2)

# เปลี่ยน age - native.country เป็น attribute ส่วน income เป็น class
# จัดการข้อมูลในส่วน training data
train_features,train_labels=split_feature_class(train_set,"income")

# ทดสอบแสดงข้อมูลออกมาดู
#print(train_features.head())
#print(train_labels.head())

# จัดการข้อมูลในส่วน test data
test_features,test_labels=split_feature_class(test_set,"income")

# สร้าง model
model=GaussianNB()
model.fit(train_features,train_labels)

# predict
clf_predict=model.predict(test_features)

# วัดผลด้วย Accuracy Score # ได้ประมาณ 79% เพราะว่ามี missing data 80-85% ถึงจะถือว่าเป็น dataset ที่ดี
print("Accuracy Score = ",accuracy_score(test_labels,clf_predict)*100)

