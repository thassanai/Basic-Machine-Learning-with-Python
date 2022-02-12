from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
## import เพิ่มเข้ามา
from sklearn.model_selection import cross_val_score
### import confusion matrix
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
import itertools
#### Accuracy
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

# สร้าง function หลังจาก training ข้อมูล
def displayImage(x):
    plt.imshow(x.reshape(28,28), # ให้เอาภาพที่ตำแหน่ง x มาแสดงผลขนาด 28x28
    cmap=plt.cm.binary,
    interpolation="nearest"
    )
    plt.show()

# ฟังก์ชันแสดงผลการทำนาย
def displayPredict(clf,actually_y,x):
    print("Actually = ",actually_y)
    print("Prediction = ",clf.predict([x])[0])

# confusion matrix
def displayConfusionMatrix(cm,cmap=plt.cm.GnBu):
    classes=["Other Number","Number 0"]
    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title("Confusion Matrix")
    plt.colorbar()
    trick_marks=np.arange(len(classes))
    plt.xticks(trick_marks,classes)
    plt.yticks(trick_marks,classes)
    thresh=cm.max()/2
    for i , j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i,format(cm[i,j],'d'),
        horizontalalignment='center',
        color='white' if cm[i,j]>thresh else 'black')

    plt.tight_layout()
    plt.ylabel('Actually')
    plt.xlabel('Prediction')
    plt.show()

mnist_raw = loadmat("/Users/thassanai/Desktop/Introduction_to_Python/Data Set/mnist-original.mat")
# สร้าง dictionary เก็บข้อมูลตัวเลข
mnist={
    "data":mnist_raw["data"].T, # ข้อมูล 70,000 ชุด
    "target":mnist_raw["label"][0] # เก็บช้อมูลกลุ่มตัวเลข 0-9
}

# ลองแสดงขนาดออกมาดู
#print(mnist["data"].shape)
#print(mnist["target"].shape)

x,y = mnist["data"], mnist["target"]

# Prepare training set , test set
# 1-600000 => training set, 600001 - 700000 => test set
x_train , x_test, y_train, y_test = x[:60000],x[60000:],y[:60000],y[60000:]

# ลองแสดงขนาดออกมาดู
#print(x_train.shape)
#print(x_test.shape)
#print(y_train.shape)
#print(y_test.shape)

# class ข้อมูล 0-9
# ข้อมูลที่ 5000 เป็นเลข 5 จริงหรือเปล่า (true,false)
# แบ่งข้อมูลออกเป็น 2 กลุ่ม
# ข้อมูลตำแหน่งที่ 5000 ค่า -> model -> class 0 หรือไม่ ? true : false
# y_train = [0,0,0....9,9,9]
pos_predict = 500 # ตอนแรกใส่ 5000 ครั้งที่ 2 ใส่ 1000 ครั้งที่ 3 ใส่ 500
y_train_0 = (y_train==0)
y_test_0 = (y_test==0)

# y_train = [true,true,true...false,false,false]

sgd_clf = SGDClassifier()
sgd_clf.fit(x_train,y_train_0) # ไปสร้าง function ข้างบน

##displayPredict(sgd_clf,y_test_0[pos_predict],x_test[pos_predict])
##displayImage(x_test[pos_predict]) # ทดสอบแสดงข้อมูล ตำแหน่งที่ 5000,1000,500

###score = cross_val_score(sgd_clf,x_train,y_train_0,cv=3,scoring="accuracy") # cv คือ จำนวนการทดลอง
###print(score)
y_train_predict = cross_val_predict(sgd_clf,x_train,y_train_0,cv=3)
cm = confusion_matrix(y_train_0,y_train_predict)

#print(cm)
#plt.figure()
#displayConfusionMatrix(cm)
y_test_predict = sgd_clf.predict(x_test)

classes = ['Other Number','Number 0']
print(classification_report(y_test_0,y_test_predict,target_names=classes))
print("Accuracy Score = ",accuracy_score(y_test_0,y_test_predict)*100)
