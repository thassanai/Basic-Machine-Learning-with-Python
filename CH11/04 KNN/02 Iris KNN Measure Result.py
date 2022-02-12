from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
# เพื่อใช้ KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,accuracy_score

iris_dataset = load_iris()

# 60-40 x จะเก็บข้อมูล feature y จะเก็บ class
x_train,x_test,y_train,y_test=train_test_split(iris_dataset['data'],iris_dataset['target'],test_size=0.4,random_state=0)

# ทดสอบแสดงผลข้อมูลออกมาดู
#print(x_train)
#print(y_train)

# สร้าง model
knn = KNeighborsClassifier(n_neighbors=5)

# training model
knn.fit(x_train,y_train)

# ทดสอบเฉลยก่อน
#print(x_test[1])
#print(y_test[1]) # กลุ่มสายพันธ์ 1 versicolor

# prediction
#y_predict = knn.predict([x_test[4]]) # ลองเปลี่ยน x_test ดู
y_predict = knn.predict(x_test) # ลองเปลี่ยน x_test ดู

# วัดผลของ model # ลองทดสอบเปลี่ยนจุดใกล้เคียงเพิ่มขึ้นดู จะเห็นว่าได้ความแม่นยำมากขึ้น
print(classification_report(y_test,y_predict,target_names=iris_dataset['target_names']))
print("Accuracy Score(%) =",accuracy_score(y_test,y_predict)*100)
