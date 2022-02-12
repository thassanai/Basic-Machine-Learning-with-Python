# if ซ้อน if
# ม. ต้น 13-14
# อายุ < 15 ม.3
# อายุ ==14 ม.2
# อายุ ==13 ม.1
# ถ้าไม่ใช่ ม.ปลาย

age=int(input("ป้อนอายุของคุณ : "))

if age<=15:
    if age==15:
        print("ม.3")
    elif age==14:
        print("ม.2")
    elif age==13:
        print("ม.1")
    else :
        print("ประถมศึกษา")
else :
    print("ม.ปลาย")

print("จบโปรแกรม")