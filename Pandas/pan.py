# Import the pandas library
import pandas as pd

# Create a DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, 30, 35, 40],
    'City': ['New York', 'Los Angeles', 'Chicago', 'Houston']
}

df = pd.DataFrame(data)

# Display the DataFrame
print("Original DataFrame:")
print(df)

# Add a new column
df['Salary'] = [70000, 80000, 120000, 95000]

# Display the updated DataFrame
print("\nDataFrame with Salary Column:")
print(df)

# Calculate the average salary
average_salary = df['Salary'].mean()
print(f"\nAverage Salary: ${average_salary:.2f}")

# Filter DataFrame for people older than 30
filtered_df = df[df['Age'] > 30]

# Display the filtered DataFrame
print("\nFiltered DataFrame (Age > 30):")
print(filtered_df)
