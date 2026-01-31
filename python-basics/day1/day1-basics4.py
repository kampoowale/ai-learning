def getEvenNumbers(nums):

    evenNumbers = []
    for n in nums:
        if n % 2 == 0:
            evenNumbers.append(n)

    return evenNumbers


nums = list(
    map(int, input("Please enter list of numbers separated by space : ").split()))

evenNumbersList = getEvenNumbers(nums)

print("The even numbers are ", evenNumbersList)
