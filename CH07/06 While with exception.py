### While ทำงานร่วมกับ Exception
while True:
    try:
        number1=int(input("ป้อนตัวเลข 1 :"))
        number2=int(input("ป้อนตัวเลข 2 :"))
        if number1 == 0 and number2 ==0:
            break
        result = number1/number2 
        # ใส่เป็น string (value error)
        # หารด้วย 0 (zero division error)
        print(result)
    except ValueError:
        print("กรุณาใส่ค่าเป็นตัวเลขเท่านั้น")
    except ZeroDivisionError:
        print("ตัวหารเป็นค่า 0 ไม่ได้")
    except Exception as e:
        print(e)
    finally :
        print("ทำงานต่อไป...(:")
