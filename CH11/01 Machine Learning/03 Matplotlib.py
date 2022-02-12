import pylab # pip3 install matplotlib
from sklearn import datasets

digit_dataset = datasets.load_digits()

print(digit_dataset.target[0]) # ดูชื่อของตัวเลข
pylab.imshow(digit_dataset.images[0],cmap=pylab.cm.gray_r)
pylab.show()
