# Read numbers from user
nums = list(map(int, input("Enter numbers separated by space: ").split()))

# Calculate sum, count, average
total = sum(nums)
count = len(nums)
average = total / count

# Print results
print("Total:", total)
print("Count:", count)
print("Average:", average)
