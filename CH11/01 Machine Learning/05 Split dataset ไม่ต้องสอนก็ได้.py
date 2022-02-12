from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris_dataset = load_iris()
print(iris_dataset.data.shape) # ดูขนาด iris datasets

# traning set 112 คิดที่ 75% ของ 150
# test set = 38 เอา 112 ลบ 150

