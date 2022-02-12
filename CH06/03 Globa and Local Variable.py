### Global Variable / Local Variable
def displayNumber():
    x=50
    a=100
    print("Hello = ",a," ใน")

### โปรแกรมหลัก Main Program
a=200
displayNumber()
print("นอก = ",a)

# Global variable คือ ตัวแปรที่ทุกส่วนสามารถเข้าถึงตัวแปรนั้นได้
# Local variable คือ ตัวแปรที่ทำงานภายในส่วน function แล้วใช้ได้แค่ในส่วนนั้นๆ เท่านั้น
# นาย x ดารา => x ทั่วประเทศรู้จัก (Global)
# นาย x หมู่บ้านแสนสุข => x ในหมู่บ้าน (Local)