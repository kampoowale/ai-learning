nums = list(map(int, input("Please enter numbers separated by space").split()))

positive = 0
negative = 0
zero = 0
for num in nums:
    if (num > 0):
        positive += 1
    elif (num < 0):
        negative += 1
    else:
        zero += 1

print("Positive", positive)
print("Negative", negative)
print("Zero", zero)
