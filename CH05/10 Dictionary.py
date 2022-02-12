### Non Primitive datatypes : Dictionary
# เราสามารถกำหนด index ได้เอง เรียกว่า key
# list , tuple
#color_list = ["สีแดง","สีเหลือง","สีเขียว"]
#color_tuple = ("สีแดง","สีเหลือง","สีเขียว")
#print(color_list[0],color_tuple[0])

### Dictionary => key (การเข้าถึงข้อมูล), value (ค่าของข้อมูล)
# index ใน list กับ tuple เป็น int เท่านั้น
# key,value สามารถกำหนดเป็น int,str,boolean,float

# การสร้าง dictionary
colors={
    "red":"สีแดง",
    "yellow":"สีเหลือง",
    "green":"สีเขียว"
}
#ระบุ key จึงจะได้ value มาใช้
#print(colors["green"])

# Construtor
pets = dict(
cat="แมว",
dog="สุนัข",
snake="งู",
bird="เบิร์ด"
)
#print(pets["bird"])
#print(pets.get("dog"))

### การแก้ไขข้อมูลใน Dictionary
#print(colors)
#colors.update({"black":"สีดำ"}) #ข้อมูลใหม่
#colors.update({"red":"สีแด๊งแดง"}) #อัพเดทข้อมูลที่มีอยู่
#print(colors)

### การใช้ loop ดึงข้อมูล colors.values() colors.keys()
#for item in colors.values():
#    print(item)

# แยก keys และ values
#for k,v in colors.items():
#    print("keys=>",k," values=>",v)

### การลบ keys
#print(colors)
#colors.pop("red")
#print(colors)

### การใช้ popitems เอาข้อมูลตัวล่าสุดที่เพิ่ง update เข้าไปออก
#print(colors)
#colors.update({"black":"สีดำ"})
#print(colors)
#colors.popitem()
#print(colors)

### clear สมาชิกทั้งหมด,ถ้าลบตัวแปรทิ้ง del colors
#print(colors)
#colors.clear()
#print(colors)

### การ copy dictionary
#x = colors.copy()
#print(x)

### การสร้าง dictionary ซ้อน dictionary
market={
    "ร้านป้าพร":{
        "desc":"ขายอาหารตามสั่ง",
        "name":"ป้าพร",
        "menu":["กะเพราไก่","ก๋วยเตี๋ยว","ข้าวผัดหมู"],
        "zone":'ตะวันออก'
    },
    "ร้านลุงป้อม":{
        "desc":"ขายผลไม้",
        "name":"น้าจ๊อบ",
        "menu":["มะม่วง","ทุเรียน"],
        "zone":'ทางเข้าตลาด'
    },
    "ร้านน้าใจ":{
        "desc":"ขายเครื่องดื่ม",
        "name":"น้าใจ",
        "menu":["นมปั่น","ชาเย็น"],
        "zone":'ข้าง 7-11'
    }
}

### การเข้าถึงข้อมูล
print(market["ร้านป้าพร"]["menu"])

### ตรวจสอบข้อมูล
#if "ทุเรียน" in market["ร้านป้าพร"]["menu"]:
#    print("เป็น")
#else :
#    print("ไม่เป็น")
