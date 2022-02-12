from sklearn import datasets
# pip3 install scikit-learn

# ndarray 
iris_dataset = datasets.load_iris()

# print keys ดู โครงสร้างที่เก็บใน iris datasets
print(iris_dataset.keys())

print(iris_dataset['target_names']) # ชื่อสายพันธุ์

print(iris_dataset['DESCR']) # คำอธิบาย
