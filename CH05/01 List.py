#a0 = 14
#a1 = 29
#a2 = 43
#a3 = 32
#a4 = 56
#a5 = 72
#a6 = 55
#a7 = 88
#print(a0+a1+a2+a3+a4+a5+a6+a7)

### Non Primitive datatypes : list
number = [] # empty list
number = [1,2,3,4,5,6,7,8,9,10,9,9,9,9,9,9] # list number มีสมาชิก 2 ตัว 1-6
name = ["alpha","bravo","charlie","delta","echo"]
mix_data = [1,"alpha",True,-88,3.14]
fruit = ["แอปเปิ้ล","มะละกอ","กล้วย","ส้ม","ทุเรียน","องุ่น"]

### Constructor list
name = list(["alpha","bravo","charlie","delta","echo"])

### การแสดงข้อมูล จัดการ ไปดูที่ power point
#print(name)

### การเข้าถึง list
#print(number[2]+number[0])
#print(name[3]+name[0])
#print(name[6])
#print(mix_data[1:4]) #ตำแหน่ง 1 ก่อนถึงตำแหน่งที่ 3

### ใช้ loop for ในการเข้าถึง list
#for x in name:
#    print(x)

### หาผลรวมจาก list
#sum = 0
#for x in number:
#    sum+=x
#print(sum)

### การแก้ไขข้อมูลใน list
#print("ก่อนแก้ไข ",number)
#number[2] = 9
#print("หลังแก้ไข ",number)

### การตรวจสอบข้อมูลใน list
#if 8 in number:
#    print("เป็นสมาชิกใน list")
#else:
#    print("ไม่เป็นสมาชิกใน list")

### การนับจำนวนสมาชิกใน list
#print(len(number))

### การประยุกต์ใช้ len ในการแสดงผลใน for loop
#for i in range(len(fruit)):
#    print(fruit[i])

#for item in fruit:
#    print(item)

### การสร้างและเพิ่มข้อมูล
### append นำสมาชิกใหม่มาต่อท้าย
#print("ก่อนเพิ่ม =>",fruit)
#fruit.append("ส้มโอ")
#print("หลังเพิ่ม =>",fruit)

### insert เพิ่มข้อมูลแบบแทรก
#print("ก่อนเพิ่ม =>",fruit)
#fruit.insert(3,"แตงโม")
#print("หลังเพิ่ม =>",fruit)

### ลบข้อมูลออกจาก list
# remove
#print("ก่อน remove =>",fruit)
#fruit.remove("มะละกอ")
#print("หลัง remove =>",fruit)

# pop การเอาสมาชิกตัวล่าสุดออกไป
#print("ก่อน pop =>",fruit)
#fruit.pop()
#print("ก่อน pop =>",fruit)

# del การลบสมาชิกลำดับที่จะลบ
#print("ก่อน del =>",fruit)
#del fruit[3]
#print("ก่อน del =>",fruit)

# clear ลบสมาชิกออก
#fruit.clear()
#print(fruit)

### คัดลอกข้อมูลใน list
#stock = []
#print(stock)
#stock = fruit.copy()
#print(stock)

### การรวมกลุ่ม list
#all = fruit + number
#print(all)

### การนับจำนวนสมาชิก count
#i = number.count(9)
#print(i)

### การหาสมาชิกใน list ที่ตำแหน่งที่เจอ
#x = number.index(5)
#print(x)

### การเรียงลำดับใน list
#print("List ก่อน sort",number)
#number.sort()
#print("List หลังจาก sort",number)
# Sort String
#print("List ก่อน sort",fruit)
#fruit.sort()
#print("List หลังจาก sort",fruit)

### เรียงลำดับชื่อ แล้ว reverse กลับ
#student=["สมพร","แก้ว","จอมขวัญ","อัมพร","ก้อง","กล้า"]
#print(student)
#student.sort()
#print(student)
#student.reverse() 
# หรือใช้แบบนี้ก็ได้ 
#student=student[::-1]
#print(student)

### การหาผลรวมใน list ที่เป็น int
print(number)
print("ค่าที่น้อยที่สุด คือ ",min(number))
print("ค่าที่มากที่สุด คือ ",max(number))
print("ผลรวม คือ ",sum(number))