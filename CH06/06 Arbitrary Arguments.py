### Arbitrary Arguments *agrs จำนวน parameter ที่กำหนดค่าไม่ได้
### *args ไม่จำกัดจำนวน ข้อมูลอยู่ในรูปแบบ tuple
def add_two(x,y):
    print(x+y)

def add(*number):
    sum=0
    for i in range(len(number)):
        sum+=number[i]
    print(sum)

#add_two(90,12,13)

add(10,5)
add(10,5,6)

### ส่ง list, tuple เข้าไปทำงานกับ function
def displayFruit(item):
     for i in range(len(item)):
         print("ผลไม้ชิ้นที่ ",i+1 ," = " , item[i])

def displayVet(item):
    for i in range(len(item)):
         print("ผักชนิดที่ ",i+1 ," = " , item[i])

fruit=["มะม่วง","มะละกอ","ฝรั่ง","มะนาว"]
vet=('ชะอม','ผักกาด','คะน้า')

displayFruit(fruit)
displayVet(vet)