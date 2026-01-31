nums = list(map(int, input("Enter numbers seperated by space").split()))

total = 0
for num in nums:
    total += num*num

print("Total of squres of entered numbers ", total)
