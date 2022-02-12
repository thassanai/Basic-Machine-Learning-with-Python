from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

iris_dataset=load_iris()

# เก็บ attribute พื้นฐานเก็บในตัวแปร x และเก็บ target ให้กับตัวแปร y
x=iris_dataset['data']
y=iris_dataset['target']

# ทดสอบแสดงข้อมูลออกมาดู
#print(x)

# prepare data 60-40
x_train,x_test,y_train,y_test=train_test_split(x,y)

# สร้าง model
nb_model=GaussianNB()

# train data
nb_model.fit(x_train,y_train)

# prediction
y_predict=nb_model.predict(x_test)

# Accuracy Score
print("Accuracy Score = ",accuracy_score(y_test,y_predict)*100)
