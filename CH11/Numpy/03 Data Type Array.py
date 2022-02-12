import numpy as np
# Datatype in Array
a = np.array([1,2,3,4,5])
a.dtype

# หรือจะกำหนดแบบนี้ก็ได้
a = np.array([1,2,3,4,5],dtype=int)


b = np.array([1.1,2.3,3.5,4.7,5.1])

c = np.zeros(5) # 0 matrix
c

d = np.zeros([2,2])
d

d = np.ones([3,3],dtype=int)

d = np.full((5,5),8,dtype=int)
d
