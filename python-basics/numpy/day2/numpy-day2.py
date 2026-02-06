import numpy as np

array = np.array(list(
    map(int, input("Please enter numbers separated by space : ").split())))

print("Total numbers enters : ", array.size)

print("Sum of all numbers : ", array.sum())

print("Average : ", array.mean())

print("Min : ", array.min(), "Max : ", array.max())

print("Devisible by 3", array[array % 3 == 0])

array[array < 0] = 0
print("Replace negative numbers with zero : ", array)
