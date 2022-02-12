### Connect MYSQL
import mysql.connector

try:
    connection = mysql.connector.connect(
        host="10.100.100.197",
        user="python",
        password="9042thas",
        database="python"
    )

    mycursor = connection.cursor()
    sql = "select * from student_info"
    mycursor.execute(sql)
    myresult = mycursor.fetchall()

    for x in myresult:
        print(x)

    ## ส่วนแรกกรอกข้อมูลนักศึกษา
    while True:
        student_id = input("กรุณาใส่รหัสนักเรียน :")
        if student_id == "exit":
            break
        print("นักเรียนรหัส [",student_id,"]")
        fname = input("กรุณาใส่ชื่อนักศึกษา: ")
        lname = input("กรุณาใส่นามสกุลนักศึกษา: ")
        score = input("กรุณากรอกคะแนนสอบ: ")
        sql_insert = "INSERT INTO student_info (student_id, student_fname, student_lname, score) VALUES (%s,%s,%s,%s)"
        insert_val = (student_id,fname,lname,score)
        mycursor.execute(sql_insert, insert_val)
        connection.commit()

    ### ดึงข้อมูลคะแนนมาจาก database
    mycursor = connection.cursor()
    sql_select = "select * from student_info"
    mycursor.execute(sql_select)
    student_info = mycursor.fetchall()
    grade = ""
    for row in student_info:
        print(row)
        student_id = row[0]
        fname = row[1]
        lname = row[2]
        score = row[3]
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
        
        # print data for check, before insert into db
        print("รหัส=",student_id,"ชื่อ-นามสกุล: ",fname," ",lname,", คะแนน=",score," เกรด=",grade)

        ### update into database
        sql_update = "update student_info set grade=%s where student_info.student_id=%s"
        update_val = (grade,student_id)
        mycursor.execute(sql_update, update_val)
        connection.commit()

except Exception as e:
    print(e)
finally:
    if connection.is_connected():
        connection.close()
        print("MySQL connection is closed")