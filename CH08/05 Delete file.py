### การลบ text file
import os

try:
    if os.path.exists("testdel.txt"):
        os.remove("testdel.txt")
        print("Deleted already!")
    else:
        print("file not found!")
except Exception as e:
    print(e)
