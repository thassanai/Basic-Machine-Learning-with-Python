# การสร้าง array
import numpy as np

arr = np.array(4) # array 0 dimension
arr.ndim #ดู dimension ของ array

# การสร้าง array 1 มิติ
a = np.array([1,2,3,4])

# การนำ list ใส่ใน array
li = [1,2,3,4]
b = np.array(li)
b

# การนำ tuple ใส่ใน array
tup01 = (1,2,3,4,5,6,7,8)
c = np.array(tup01)
c

# การสร้าง array 2 มิติ
d = np.array([[1,2,3],[4,5,6]])
d

lis01 = [[1,2,3],[4,5,6],[7,8,9]]
e = np.array(lis01)
e

tup02 = ((1,2,3)(4,5,6),(7,8,9))
f = np.array(tup02)
f

# การสร้าง array 3 มิติ
lis02 = [[[1,2,3],[11,12,13]]] # 1 ชั้น
lis03 = [[[1,2,3],[11,12,13]],[[[4,5,6],[21,22,23]]]] # 2 ชั้น ดูจากจำนวน bucket

