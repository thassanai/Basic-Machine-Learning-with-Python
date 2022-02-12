#การเข้าถึงสมาชิกใน array
import numpy as np

a = np.array([1,2,3,4,5])
a

a[2]
a[-1]
a[0]

# แก้ไขค่าใน array
a[2] = 88
a

# 2 dimension
b = np.array([[1,2],[3,4],[5,6]])
b[1][1]
b[2][1]
