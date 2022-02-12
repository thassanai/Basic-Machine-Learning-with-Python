### การสร้างไฟล์ใหม่
try:
    file = open("newfile.txt","w",encoding="utf-8")
    file.write("เขียนลงไฟล์ hello world")
    file.close()
except FileNotFoundError:
    print("หาไฟล์ไม่เจอ, File not found!")