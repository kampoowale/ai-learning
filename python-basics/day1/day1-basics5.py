def get_even_numbers(nums):
    evenNums = []

    for n in nums:
        if n % 2 == 0:
            evenNums.append(n)

    return [n for n in nums if n % 2 == 0]
# [item for item in iterable if condition]


def get_odd_nums(nums):
    oddNums = []

    for n in nums:
        if (n % 2 != 0):
            oddNums.append(n)

    return oddNums


def count_positive_negative_zero(nums):

    positive = 0
    negative = 0
    zero = 0

    for n in nums:
        if (n > 0):
            positive += 1

        elif (n < 0):
            negative += 1

        else:
            zero += 1

    return positive, negative, zero


def get_squres(nums):

    return [n*n for n in nums]
# [expression for item in iterable]


userNumbers = list(
    map(int, input("Please enter numbers separated by space").split()))

evenNumbers = get_even_numbers(userNumbers)
oddNumbers = get_odd_nums(userNumbers)
pos, neg, zero = count_positive_negative_zero(userNumbers)
squres = get_squres(userNumbers)

print("Even Numbers : ", evenNumbers)
print("Odd Numbers : ", oddNumbers)
print("Total positive negative and zeros : ", pos, neg, zero)
print("Squeres : ", squres)
