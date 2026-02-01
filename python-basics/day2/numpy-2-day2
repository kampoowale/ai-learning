import numpy as np

array = np.array(
    list(map(int, input("Please enter numbers separated by space : ").split())))

print("You have entered : ", array)

print("Count : ", array.size)

print("Sum : ", array.sum())

print("Mean : ", array.mean())

print("Standard deviation : ", array.std())

print("Numbers greater than mean", array[array > array.mean()])

print("Numbers between 5 to 20 (inclusive)",
      array[(array >= 5) & (array <= 20)])

print("Not devisible by 2", array[array % 2 != 0])

odd = array[array % 2 != 0]
print("Squres only the odd : ", odd * odd)

array[array > 50] = 50
print("Replace greater than 50 with 50 : ", array)
