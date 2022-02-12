from sklearn import datasets

digit_dataset = datasets.load_digits()

print(digit_dataset.keys())
print(digit_dataset['DESCR'])
print(digit_dataset['data']) # ดูข้อมูล
print(digit_dataset['target_names']) # ดูว่าเก็บตัวเลขอะไรบ้าง
print(digit_dataset.images[0]) # ดูข้อมูล dataset ในรูปแรก
print(digit_dataset.images[0].shape) # ดูขนาดของ arrays
