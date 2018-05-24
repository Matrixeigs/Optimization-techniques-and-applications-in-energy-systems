from numpy import array
from scipy import matrix

a = array([[1, 0, -1, 0], [1, 0, 0, -1]])
b = matrix([1, 1, 0, 0]).transpose()
c = matrix([[1, 0, -1, 0], [1, 0, 0, -1]])
d = c * b
e = matrix(a) * b
print(d[0])
