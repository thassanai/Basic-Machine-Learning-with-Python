### โปรแกรมแบ่งเป็น 2 ส่วน
# ส่วนแรก: จะเป็นการรับข้อมูลนักเรียน รหัสนักเรียน และคะแนนสอบ
# ส่วนที่สอง: จะเป็นการอ่านข้อมูลไฟล์ข้อมูลนักเรียนในส่วนแรกมาตัดเกรด
try:
    ### ส่วนแรก: ส่วนของการรับข้อมูลนักเรียน
    fscore=open("score.txt","a",encoding="utf-8")
    while True:
         student_id = input("กรุณาใส่รหัสนักเรียน :")
         if student_id == "exit":
             break
         print("นักเรียนรหัส [",student_id,"]")
         score = input("กรุณากรอกคะแนนสอบ: ")
         fscore.writelines(student_id+"\t"+score+"\n")
    fscore.close()

    ### ส่วนที่สอง: อ่านข้อมูลจากไฟล์ score.txt มาตัดเกรดแล้วบันทึก
    fscore=open("score.txt","r",encoding="utf-8")
    fgrade=open("grade.txt","w",encoding="utf-8")
    fgrade.writelines("รหัสนักเรียน"+"\t"+"คะแนน"+"\t"+"เกรด"+"\n")
    grade = ""
    for line in fscore.readlines():
        print(line)
        student_id,score = line.split("\t")
        print("รหัส=",student_id,", คะแนน=",score)
        score = int(score)
        if score>=80:
            grade="A"
        elif score>=70 and score<80:
            grade="B"
        elif score>=60 and score<=69:
            grade="C"
        elif score>=50 and score<=59:
            grade="D"
        else:
            grade="F"
        fgrade.writelines(student_id+"\t"+str(score)+"\t"+grade+"\n")
    fscore.close()
    fgrade.close()
    
except Exception as e:
    print(e)