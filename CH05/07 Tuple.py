### Non Primitive datatypes : tuple
number = (1,2,3,4,5,6,7,8,9,10,9,9,9,9,9,9) # list number มีสมาชิก 2 ตัว 1-6
name = ("alpha","bravo","charlie","delta","echo")
fruit = ("แอปเปิ้ล","มะละกอ","กล้วย","ส้ม","ทุเรียน","องุ่น")

### Constructor list
example_tuple = tuple((1,"alpha",True,-88,3.14))
example_list = list([1,"alpha",True,-88,3.14])

### การแสดงข้อมูล จัดการ ไปดูที่ power point
#example_list[2] = "เขาค้อ"
#example_tuple[2] = "เขาค้อ"
#print(example_tuple)
#print(example_list)

### การเข้าถึง tuple
#c_list = list(example_tuple)
#c_list[1] = "echo"
#print(c_list)
#f_tuple = tuple(c_list)
#print(f_tuple)

### ใช้ loop for ในการเข้าถึง tuple
#for item in example_tuple:
#    print(item)

### หาผลรวมจาก tuple
#sum = 0
#for x in number:
#    sum+=x
#print(sum)

### การตรวจสอบข้อมูลใน tuple
#if "มะม่วง" in fruit:
#    print("เป็นสมาชิกใน tuple")
#else:
#    print("ไม่เป็นสมาชิกใน tuple")

### การนับจำนวนสมาชิกใน tuple
#print(len(number))

### การประยุกต์ใช้ len ในการแสดงผลใน for loop
#for i in range(len(fruit)):
#    print(fruit[i])

#for item in fruit:
#    print(item)

### การเพิ่มข้อมูลใน tuple
### การ join tuple
#new = "foxtrot"
#new = ("foxtrot",) # สร้าง new ในรูปแบบ tuple เพื่อเพิ่มเข้าไปใน tuple name
#name = name+new
#name = name+(new,)# cast string 
#print(name)

### del การลบสมาชิกลำดับที่จะลบ
#print("ก่อน del =>",fruit)
#del fruit
#print("หลัง del =>",fruit)

### การนับจำนวนสมาชิก count
#i = number.count(9)
#print(i)

### การค้นหาสมาชิกใน tuble ด้วย index
#position =name.index("delta")
#print(position)

### การเรียงลำดับใน tuble ให้เปลี่ยนไปเป็น list ก่อนกลับไปดูตัวอย่าง list

### ความพิเศษของ tuple
### การนำค่า tuple ไปใส่ในตัวแปร a,b,c,d
#x = (20,30,40,50)
#a,b,c,d = x
#print(a,b,c,d)

### การสลับค่าระหว่าง tuble
#POI_01 = (13.767006,100.5925583) #JOBBKK.COM
#POI_02 = (17.6334887,100.0913382) #URU
#print("Before POI01",POI_01)
#print("Before POI02", POI_02)
#POI_01,POI_02 = POI_02,POI_01
#print("After POI01",POI_01)
#print("After POI02", POI_02)

### การเปลี่ยน tuple -> String
#myname = ('T','H','A','S','S','A','N','A','I')
#name = "".join(myname) #myname เป็น tuple ต้องมา join กับ string
#print(name) 

## การสลับตำแหน่งจากหน้าไปหลัง reverse tuple
#x = (900,30,13,89,1000,66) # reverse int
#y = tuple(reversed(x))
#print(y)

### string -> tuple & reverse
#myname = "thassanai"
#myname_tup = tuple(myname)
#print(myname_tup)
#myname_rev = tuple(reversed(myname_tup))
#print(myname_rev)