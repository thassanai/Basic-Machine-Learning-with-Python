### Factorial
import sys

def factorial(number):
    if number==1:
        return number
    else :
        return number * factorial(number-1)

x=factorial(10)
print(x)

print(sys.maxsize)
i = 10**100
print(i)
print(i > sys.maxsize)

"""
3!

3
3 x factorial(2)
3 x 2 x 1 = 6


"""

