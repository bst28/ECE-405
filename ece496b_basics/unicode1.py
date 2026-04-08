################################## Problem 1 #####################

# (a)
# chr(0) creates the null character (ASCII value 0)
chr(0)
print(chr(0))  # prints nothing visible because the null character is invisible

# (b)
# repr() shows the string in a readable form
print(repr(chr(0)))  # shows '\x00', which is how the null character is represented

# (c)
# chr(0) can be included inside a string
chr(0)
print(chr(0))  # still prints nothing visible

# Concatenate a normal string with a null character in the middle
"this is a test" + chr(0) + "string"
print("this is a test" + chr(0) + "string")  # text prints, but the null character is invisible









