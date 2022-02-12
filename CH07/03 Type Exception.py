### ValueError ค่าผิดพลาด
### ZeroDivisionError หารด้วย 0
### type อื่นๆ https://docs.python.org/3/library/exceptions.html#bltin-exceptions
try:
    number1=int(input("ป้อนตัวเลข 1 :"))
    number2=int(input("ป้อนตัวเลข 2 :"))
    result = number1/number2 
    # ใส่เป็น string (value error)
    # หารด้วย 0 (zero division error)
    print(result)
except ValueError:
    print("ต้องใส่ค่าตัวเลขเท่านั้น")
except ZeroDivisionError:
    print("หารด้วย 0 ไม่ได้")
except TypeError:
    pass #ไม่รู้จะให้ทำอะไรใส่ pass ไปก่อนได้
