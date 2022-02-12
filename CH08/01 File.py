### open("ชื่อไฟล์","โหมด","รูปแบบการเข้ารหัส")
# r=readwax, w=write เขียนทับ, a=append เขียนต่อท้าย,
# x = creat, t = textfile, b = binaryfile

file = open("test.txt","r",encoding="utf-8")
#print(file) # แสดง fileread
print(file.read()) # อ่านจากไฟล์ทั้งหมด # ถ้าอ่านภาษาไทยไม่ได้ต้องใส่ encoding="UTF-8"
#print(file.read(5)) # อ่านจากไฟล์ 5 ตัวแรก
file.close()

#เราสามารถใส่ try: except: เพื่อป้องกันการหาไฟล์ไม่เจอได้
try:
    file = open("tests.txt","r")
    print(file.read())
    file.close()
except FileNotFoundError:
    print("หาไฟล์ไม่เจอ, File not found!")
