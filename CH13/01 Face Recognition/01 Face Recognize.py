from tokenize import PlainToken
from sklearn.datasets import fetch_lfw_people # โหลด dataset
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline # สร้างตัวเชื่อมต่อระหว่าง PCA กับ SVM
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV #gridsearch
from sklearn.metrics import accuracy_score,confusion_matrix
import seaborn as sb

# Download 60 face images
faces=fetch_lfw_people(min_faces_per_person=60) # โหลด dataset มาใช้ ระบุค่าใบหน้าบุคคลน้อยที่สุด

# แสดงข้อมูลใบหน้ามาดู
#print(faces.target_names) #ชื่อเจ้าของใบหน้า
#print(faces.images.shape)

# แสดง dataset
# แสดงรูปใบหน้า พร้อมกับแสดงชื่อ 3x5
#figure,ax=plt.subplots(3,5)
# แสดงรูปใบหน้า พร้อมกับแสดงชื่อ
#for i,axi in enumerate(ax.flat):
#    axi.imshow(faces.images[i],cmap='bone')
#    axi.set(xticks=[],yticks=[])
#    axi.set_ylabel(faces.target_names[faces.target[i]].split()[-1],color='black')
#plt.show()

# reduce & create model
pca=PCA(n_components=150,svd_solver="randomized",whiten=True)
svc=SVC(kernel='rbf',class_weight='balanced')
model=make_pipeline(pca,svc) # รวมประกอบ model

# สร้างชุดข้อมูล train,test data
x_train,x_test,y_train,y_test=train_test_split(faces.data,faces.target,random_state=45)

# ระบุค่า regulalize 
param_grid={"svc__C":[1,5,10,50],"svc__gamma":[0.0001,0.0005,0.001,0.005]}

# train data to model
grid=GridSearchCV(model,param_grid)
grid.fit(x_train,y_train)
# print(grid.best_params_)
# print(grid.best_estimator_)


#print(grid.best_params_) # ค่า grid ตัวไหนที่เหมาะสม เอาไป assign ค่าใน model
model=grid.best_estimator_

# prediction
y_predict=model.predict(x_test)

# แสดงผลการทำนาย
#fig,ax=plt.subplots(4,6)
# แสดงรูปใบหน้า พร้อมกับแสดงชื่อ
#for i,axi in enumerate(ax.flat):
#    axi.imshow(x_test[i].reshape(62,47),cmap='bone')
#    axi.set(xticks=[],yticks=[])
#    axi.set_ylabel(faces.target_names[y_predict[i]].split()[-1],
#    color='green' if y_predict[i]==y_test[i] else 'red') # เอาข้อความที่พยากรณ์ออกมา y_predict ว่าตรงกันหรือเปล่า
#plt.show()

#print("Accuracy Score = ",accuracy_score(y_test,y_predict)*100)
mat=confusion_matrix(y_test,y_predict)
sb.heatmap(mat.T,square=True,annot=True,fmt='d',cbar=False,
    xticklabels=faces.target_names,
    yticklabels=faces.target_names
    )
plt.xlabel("True Data")
plt.ylabel("Predict Data")
plt.show()
