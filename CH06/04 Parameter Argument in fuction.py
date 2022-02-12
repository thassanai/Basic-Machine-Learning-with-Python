### Function Structor, Parameter/Argument
# การรับที่ส่งค่าเข้ามาในฟังก์ชัน

def myData(firstname,lastname,age):
    print("ชื่อ = ",firstname," นามสกุล = ",lastname,"อายุ = ",age)

def addition(a,b):
    print("ผลรวม = ",a+b)

myfname="Thassanai"
mylname="Mhuansean"
myage=25
myData(myfname,mylname,myage)

x=5
y=9
addition(x,y)

# Arguments => ค่าที่ส่งเข้าไปใน function => myfname , mylname , myage (ตอนที่เรียกใช้ฟังก์ชั่น)
# Parameter => ค่าตัวแปรที่รับข้อมูลที่ส่งมาทำงาน (Arguments) => firstname , lastname , age

# อาส่ง -  พารับ
