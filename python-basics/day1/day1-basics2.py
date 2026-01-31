nums = list(
    map(int, input("Enter a list of numbers separated by space ").split()))

even = 0
odd = 0
for num in nums:

    if (num % 2 == 0):
        even += 1
    else:
        odd += 1

print("Total even number you entered are", even)
print("Total odd number you entered are", odd)
