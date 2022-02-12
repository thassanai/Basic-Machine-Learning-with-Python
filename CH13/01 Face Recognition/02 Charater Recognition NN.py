from configparser import Interpolation
from random import shuffle
from re import M
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

mnist_raw=loadmat("/Users/thassanai/Desktop/Introduction_to_Python/Data Set/mnist-original.mat")
mnist={
    "data":mnist_raw["data"].T,
    "target":mnist_raw["label"][0]
}

# train data, test data 80-20
x,y=mnist["data"],mnist["target"]
# shuffle data
sf=np.random.permutation(70000)
x,y=x[sf],y[sf]
x_train,x_test,y_train,y_test=x[:60000],x[60000:],y[:60000],y[60000:]

## แสดงขนาดของ train data และ test data
#print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

# fig,ax=plt.subplots(10,10,
# figsize=(8,8),
# subplot_kw={'xticks':[],'yticks':[]},
# gridspec_kw=dict(hspace=0.1,wspace=0.1))

## Display images data before training
# for i,axi in enumerate(ax.flat):
#     axi.imshow(x_train[i].reshape(28,28),cmap='binary',interpolation='nearest')
#     axi.text(0.05,0.05,str(int(y_train[i])),transform=axi.transAxes,color="black")
# plt.show()

## Create model
model=MLPClassifier()
model.fit(x_train,y_train)

# Prediction
y_predict=model.predict(x_test)

# แสดง Accuracy Score
# print("Accuracy Score = ",accuracy_score(y_test,y_predict)*100)

fig,ax=plt.subplots(10,10,
figsize=(8,8),
subplot_kw={'xticks':[],'yticks':[]},
gridspec_kw=dict(hspace=0.1,wspace=0.1))

## Display images data after training and prediction
for i,axi in enumerate(ax.flat):
    # Display test image data
    axi.imshow(x_test[i].reshape(28,28),cmap='binary',interpolation='nearest')
    # Display test number image data
    axi.text(0.05,0.05,str(int(y_test[i])),transform=axi.transAxes,color="black")
    # Display predict number image data
    axi.text(0.75,0.05,str(int(y_predict[i])),transform=axi.transAxes,
    color="green" if y_predict[i]==y_test[i] else "red")
plt.show()
