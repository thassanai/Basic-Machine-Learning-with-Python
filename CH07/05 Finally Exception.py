### Finally มี/ไม่มีก็จะทำงาน
try:
    number1=int(input("ป้อนตัวเลข 1 :"))
    number2=int(input("ป้อนตัวเลข 2 :"))
    result = number1/number2 
    # ใส่เป็น string (value error)
    # หารด้วย 0 (zero division error)
    print(result)
except Exception as e:
    print(e)
finally :
    print("ทำงานต่อไป...(:")
