### อ่านค่าจากไฟล์ทีละบรรทัด โดยใช้ loop for

try:
    file=open("test.txt","a",encoding="utf-8")
    file.writelines("สวัสดีปีใหม่\n") #เขียนข้อมูลต่อท้ายทีละบรรทัด
    for i in range(3):
        name = input("ใส่ข้อความที่ต้องการ :")
        file.writelines(name+"\n")
    file.close()
except FileNotFoundError:
    print("หาไฟล์ไม่เจอ, File not found!")
