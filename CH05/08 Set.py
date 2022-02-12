### Non Primitive datatypes : Set
#แบบปกติ
fruit={"มะขาม","มะม่วง","มะปราง","มะยม"}
#print(fruit)

# สร้างแบบ constructor
#fish=set(["ปลาช่อน","ปลานิล","ปลาตะเพียน"])
#print(fish)

# สร้างเป็น list ก่อน
#pet = ("สุนัข","แมว","ปลาทอง","กระรอก")
#pet_set = set(pet)
#print(pet_set)

# ข้อมูลแบบผสม
#mix_data = {"มะม่วง",99,True,-199}
#print(mix_data)

### เพิ่มข้อมูลสมาชิกใน Set
#fruit.add("ทุเรียน")
#fruit.add("สับปะรด")
#fruit.add(999)
#print(fruit)

#เพิ่มสมาชิกหลายตัวใน Set
#fruit_add = ("ตะไคร้","โหระพา","ชะอม")
#fruit.update(fruit_add)
#fruit.update(["ตะไคร้","โหระพา","ชะอม"])
#print(fruit)

#ใช้ Loop เพื่อแสดงผลสมาชิกใน Set
#for item in fruit:
#   print("ข้อมูล :",item)

#แสดงจำนวนสมาชิกใน set
#print(len(fruit))

# ตรวจสอบข้อมูลใน Set
#if "มะเฟือง" in fruit:
#    print("มีข้อมูลใน set")
#else:
#    print("ไม่มีข้อมูลใน set")

# การลดรูปการตรวจสอบข้อมูล
#print("มะเฟือง" in fruit)

# การลบข้อมูลใน Set
#print(fruit)
#fruit.remove("มะม่วง")
#print(fruit)

# การลบข้อมูลแบบไม่แสดง error
#fruit.discard("ทุเรียน") # remove / discard
#print(fruit)

# การลบข้อมูลทั้งหมดใน Set
#fruit.clear()
#del fruit ลบตัวแปร
#print(fruit)