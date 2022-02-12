### **kwargs เป็นการกำหนดชื่อ parameter ใช้งานใน function ได้ไม่จำกัด
### จะส่งข้อมูลเข้ามาในรูปแบบ dictionary
# *args => ค่าใน parameter มีได้หลายค่า

def add(*number):
    sum=0
    for i in range(len(number)):
        sum+=number[i]
    print(sum)

def displayData(fname):
    print(fname)

### ชื่อ Parameter มีได้หลายชื่อตัวแปร
def displayData(**item):
    print(item)

displayData(fname="Thassanai")
displayData(fname="Thassanai",lname="Mhuansean")
displayData(fname="Thassanai",lname="Mhuansean",city="กรุงเทพ")
displayData(fname="Thassanai",lname="Mhuansean",status="โสด")
displayData(fname="Thassanai",lname="Mhuansean",age=20,city="กรุงเทพ",status="แต่งงาน")
displayData(fname="Thassanai",lname="Mhuansean",age=20,city="กรุงเทพ",status="แต่งงาน",job="Programmer")
displayData(fname="Thassanai",lname="Mhuansean",country="ระยอง")
