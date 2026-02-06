# Read numbers from user
nums = list(map(int, input("Enter numbers separated by space: ").split()))

# input("Enter numbers separated by space: ")
# Prompts user to enter a number

# input("Enter numbers separated by space: ").split()
# Splits the numbers entered by user by space(by default)

# ['10', '20', '30']
# (map(int, input("Enter numbers separated by space: ").split()))
# (map(int, ['10', '20', '30']))
# map(int,...) Converts the list of string into map of integer

# list(map(...))
# converts map into list

# Calculate sum, count, average
total = sum(nums)
count = len(nums)
average = total / count

# Print results
print("Total:", total)
print("Count:", count)
print("Average:", average)
