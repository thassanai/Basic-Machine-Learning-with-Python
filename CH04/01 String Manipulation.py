# การต่อ string (string concatinate)
#firstName = "Thassanai"
#lastName = "Mhuansean"
#gender = "Male"
#salary = 10081.9281

#print("Firstname : "+firstName +"\tLastname : "+ lastName+"\tGender : "+gender)

# การจัดรูปแบบการแสดงผล format
#CustomerDetail = "Firstname : "+firstName +"\tLastname : "+ lastName+"\tGender : "+gender
#print(CustomerDetail)

#CustomerDetail = "Firstname : {0}\tLastname : {1} \tGender : {2} \tSalary: {3:.2f}"

#print(CustomerDetail.format(firstName,lastName,gender,salary))

# การนับจำนวนคำในประโยค count()
#edu = "จบปริญญาตรี ที่เมืองเช็ค จบปริญญาเอก ที่ชิคาโก้ จบปริญญาโท ที่วัดหนองปลาไหล"
#print(edu.count("ปริญญา"))

# การตรวจสอบคำขึ้นต้น/คำลงท้าย
#name = "นาย ทัศไนย เหมือนเสน"
#print(name.startswith("นาย"))
#if(name.startswith("นาย")):
#    print("เพศชาย")
#else :
#    print("เพศหญิง")

# คำลงท้าย
month = "มกราคม"
if(month.endswith("คม")):
    print("เดือนนี้มี 31 วัน")
elif(month.endswith("ยน")):
    print("เดือนนี้มี 30 วัน")
else :
    print("เดือนกุมภาพันธ์มี 28/29 วัน")

