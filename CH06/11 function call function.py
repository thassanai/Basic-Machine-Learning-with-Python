### ฟังก์ชั่น เรียก ฟังก์ชั่น
def displayFood():
    print("หูฉลาม")

def displayCity():
    print("สวัสดีกรุงเทพ")
    displayFood()

displayCity()

# การเปรียบเทียบตัวเลข 3 ค่า
def findMax(x,y,z):
    return max(max(x,y),z)

def max(x,y):
    if x>y:
        return x
    if y>x:
        return y

max = findMax(98,44,43)
print("ค่าที่สูงที่สุดคือ ",max)
