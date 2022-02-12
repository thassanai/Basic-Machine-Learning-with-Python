### function return ค่า
### 1. ไม่มีการรับค่า ไม่มีการส่งค่า
### 2. มีการรับค่าเข้าไปทำงาน
### 3. ไม่มีการรับค่า แต่มีการส่งค่าออกไป
### 4. มีการรับค่าเข้าไปทำงาน มีการส่งค่าออกมา (return) เพื่อไปทำงานที่โปรแกรมหลัก
def pi():
    return 3.1415

r = 10
circle_area = pi()*r*r
print(circle_area)

def add(a,b):
    return a+b

sum=add(98,23)
print(sum)

### return เพื่อกระโดดออกจาก function
def randomNumber(x):
    if len(x)<3 :
        return
    if x == "100":
        print("ถูกหวย")
        return 4000
    else :
        print("ไม่ถูกหวย")
        return 0

money=randomNumber("100")
print("เงินรางวัล = ",money," บาท")