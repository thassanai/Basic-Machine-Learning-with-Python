### อ่านค่าจากไฟล์ทีละบรรทัด โดยใช้ loop for
try:
    file=open("test.txt","r",encoding="utf-8")
    lines=file.readlines() #อ่านข้อมูลมาแสดงทีละบรรทัด
    for line in lines:
        print(line)
    file.close()
except FileNotFoundError:
    print("หาไฟล์ไม่เจอ, File not found!")