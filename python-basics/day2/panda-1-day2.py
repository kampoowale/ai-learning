import pandas as pd

# 1️⃣ Read CSV file
# replace with full path if not in the same folder
df = pd.read_csv("data.csv")

# 2️⃣ Print the whole DataFrame
print("Full Data:")
print(df)

# 3️⃣ Average Age and Score
print("\nAverage Age:", df['Age'].mean())
print("Average Score:", df['Score'].mean())

# 4️⃣ Students with Score > 80
high_score_students = df[df['Score'] > 80]
print("\nStudents with Score > 80:")
print(high_score_students)

# 5️⃣ Add a new column "Passed" (True if Score >= 70, else False)
df['Passed'] = df['Score'] >= 70
print("\nUpdated DataFrame with 'Passed' column:")
print(df)
