### lambda function (Anonymous function)
### ฟังก์ชันที่สร้างมาแล้วไม่อยากระบุชื่อ
# lambda argument : expression
sum = lambda x,y : x+y
multiply = lambda x,y: x*y

print(sum(90,18))
print(multiply(10,18))

def myPower(x):
    # x = ตัวเลข
    # a = เลขยกกำลัง
    return lambda a : x**a

y=myPower(5)
result=y(2)
print(result)
