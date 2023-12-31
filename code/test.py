import numpy

list = [5.3, 2.4, 5.1, 3.6]

array = numpy.array(list)

print(numpy.isnan(array).any())