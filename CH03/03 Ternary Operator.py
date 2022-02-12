#Ternary Operator การลดรูปคำสั่ง if else
# "เงื่อนไขเป็นจริง" if expresion else "เงื่อนไขอื่นๆ"
age=int(input("ป้อนอายุของคุณ :"))
if age > 15:
    print("วัยรุ่น")
else:
    print("วัยเด็ก")

# เมื่อใช้ ternary operator ลดรูปก็จะได้
age=int(input("ป้อนอายุของคุณ :"))
# 15 - 20 => วัยรุ่น
print("วัยรุ่น") if age>=15 else print("วัยเด็ก")

print("จบโปรแกรม")




# and or not 