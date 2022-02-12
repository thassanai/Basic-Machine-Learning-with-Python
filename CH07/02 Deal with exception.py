# number1=int(input("ป้อนตัวเลข 1 :"))
# number2=int(input("ป้อนตัวเลข 2 :"))
# result = number1/number2 
# # ใส่เป็น string (value error)
# # หารด้วย 0 (zero division error)
# print(result)

"""
try:
    รันคำสั่งโปรแกรมปกติ
except:
    คำส่ังที่ทำงานตอนโปรแกรมมีข้อผิดพลาด

"""
try:
    number1=int(input("ป้อนตัวเลข 1 :"))
    number2=int(input("ป้อนตัวเลข 2 :"))
    result = number1/number2 
    # ใส่เป็น string (value error)
    # หารด้วย 0 (zero division error)
    print(result)
except:
    print("โปรแกรมมีข้อผิดพลาด")