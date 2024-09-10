# # # import numpy as np

# # # # # Create a NumPy array with values from 1 to 5
# # # # array = np.array([1, 2, 3, 4, 5])

# # # # # Print the original array
# # # # print("Original Array:")
# # # # print(array)

# # # # # Perform some basic operations
# # # # array_sum = np.sum(array)             # Sum of all elements
# # # # array_mean = np.mean(array)           # Mean of all elements
# # # # array_max = np.max(array)             # Maximum value
# # # # array_squared = np.power(array, 2)    # Square each element

# # # # # Print the results
# # # # print("\nBasic Operations:")
# # # # print(f"Sum: {array_sum}")
# # # # print(f"Mean: {array_mean}")
# # # # print(f"Maximum: {array_max}")
# # # # print(f"Squared Elements: {array_squared}")

# # # # array_a = np.append(array,12)
# # # # print(array_a)

# # # # import math
# # # # a = np.array([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [9, 10, 11, 12],[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [9, 10, 11, 12]]])
# # # # print(np.arange(-2, 12, 1))
# # # # print(np.linspace(0, 5, num=9))
# # # # print(a.reshape(3, 2))
# # # # a = np.array([1, 2, 3, 4])
# # # # b = np.array([5, 6, 7, 8])
# # # # print(a)
# # # # print(b)
# # # # b = a.reshape(1, 1)
# # # # print(b)
# # # import pandas

# # # a = np.array([[[1, 2, 3, 4, 52, 1],[1, 2, 3, 4, 52, 11]],[[1, 2, 3, 4, 52, 1],[1, 2, 3, 4, 52, 11]],[[1, 2, 3, 4, 52, 1],[1, 2, 3, 4, 52, 11]]])
# # # b=a.shape
# # # a2 = a[:, np.newaxis]
# # # c=a2.shape

# # # print(c)










# # # Import the pandas library
# # # import pandas as pd
# # import pandas as pd

# # # Create a DataFrame
# # data = {
# #     'Name': ['Alice', 'Bob', 'Charlie', 'David'],
# #     'Age': [25, 30, 35, 40],
# #     'City': ['New York', 'Los Angeles', 'Chicago', 'Houston']
# # }

# # df = pd.DataFrame(data)

# # # Display the DataFrame
# # print("Original DataFrame:")
# # print(df)

# # # Add a new column
# # df['Salary'] = [70000, 80000, 120000, 95000]

# # # Display the updated DataFrame
# # print("\nDataFrame with Salary Column:")
# # print(df)

# # # Calculate the average salary
# # average_salary = df['Salary'].mean()
# # print(f"\nAverage Salary: ${average_salary:.2f}")

# # # Filter DataFrame for people older than 30
# # filtered_df = df[df['Age'] > 30]

# # # Display the filtered DataFrame
# # print("\nFiltered DataFrame (Age > 30):")
# # print(filtered_df)



























# import pandas as pd
# import numpy as np

# def main():
#     # 1. MultiIndex (Hierarchical Indexing)
#     print("=== MultiIndex (Hierarchical Indexing) ===")
#     arrays = [
#         ['A', 'A', 'B', 'B'],
#         [1, 2, 1, 2]
#     ]
#     index = pd.MultiIndex.from_arrays(arrays, names=('Letter', 'Number'))
#     df_multiindex = pd.DataFrame({'Value': [10, 20, 30, 40]}, index=index)
#     print(df_multiindex)
#     print(df_multiindex.loc['A'])
#     print(df_multiindex.loc[('A', 1)])
#     print()

#     # 2. Advanced GroupBy Operations
#     print("=== Advanced GroupBy Operations ===")
#     df_groupby = pd.DataFrame({
#         'A': ['foo', 'foo', 'foo', 'bar', 'bar'],
#         'B': ['one', 'one', 'two', 'two', 'one'],
#         'C': np.random.randn(5),
#         'D': np.random.rand(5)
#     })
#     grouped = df_groupby.groupby(['A', 'B'])
    
#     # Custom Aggregation Functions
#     result = grouped.agg({
#         'C': ['mean', 'std'],
#         'D': 'sum'
#     })
#     print(result)
    
#     # Transformations
#     df_groupby['C_standardized'] = grouped['C'].transform(lambda x: (x - x.mean()) / x.std())
#     print(df_groupby)
#     print()

#     # 3. Time Series Analysis
#     print("=== Time Series Analysis ===")
#     dates = pd.date_range('2023-01-01', periods=100)
#     data = pd.Series(np.random.randn(100), index=dates)
    
#     # Resampling and Frequency Conversion
#     monthly_data = data.resample('M').mean()
#     print(monthly_data)
    
#     # Rolling Window Calculations
#     rolling_mean = data.rolling(window=7).mean()
#     print(rolling_mean)
#     print()

#     # 4. Data Merging and Joining
#     print("=== Data Merging and Joining ===")
#     df1 = pd.DataFrame({'key': ['A', 'B', 'C'], 'value1': [1, 2, 3]})
#     df2 = pd.DataFrame({'key': ['A', 'B', 'D'], 'value2': [4, 5, 6]})
    
#     # Merge with Different Join Types
#     inner_join = pd.merge(df1, df2, on='key', how='inner')
#     print(inner_join)
    
#     outer_join = pd.merge(df1, df2, on='key', how='outer')
#     print(outer_join)
    
#     # Concatenation and Appending
#     df3 = pd.DataFrame({'key': ['E', 'F'], 'value1': [7, 8]})
#     concatenated = pd.concat([df1, df3], ignore_index=True)
#     print(concatenated)
#     print()

#     # 5. Pivot Tables and Crosstabs
#     print("=== Pivot Tables and Crosstabs ===")
#     data = pd.DataFrame({
#         'Date': pd.date_range('2023-01-01', periods=6),
#         'Category': ['A', 'B', 'A', 'B', 'A', 'B'],
#         'Value': [10, 20, 15, 25, 30, 35]
#     })
    
#     # Creating Pivot Tables
#     pivot_table = pd.pivot_table(data, values='Value', index='Date', columns='Category', aggfunc='sum')
#     print(pivot_table)
    
#     # Cross Tabulations
#     crosstab = pd.crosstab(data['Category'], data['Value'])
#     print(crosstab)
#     print()

#     # 6. Handling Missing Data
#     print("=== Handling Missing Data ===")
#     df_missing = pd.DataFrame({
#         'A': [1, 2, np.nan, 4],
#         'B': [5, np.nan, np.nan, 8]
#     })
    
#     # Filling Missing Values
#     df_filled = df_missing.fillna(value={'A': 0, 'B': df_missing['B'].mean()})
#     print(df_filled)
    
#     # Interpolation
#     df_interpolated = df_missing.interpolate()
#     print(df_interpolated)
#     print()

#     # 7. Efficient Data Manipulation
#     print("=== Efficient Data Manipulation ===")
#     df_efficient = pd.DataFrame({
#         'A': [1, 2, 3, 4],
#         'B': [5, 6, 7, 8]
#     })
    
#     # Vectorized Operations
#     df_efficient['C'] = df_efficient['A'] * 2
#     print(df_efficient)
    
#     # Using `apply()` for Custom Functions
#     def custom_function(x):
#         return x.max() - x.min()
    
#     result = df_efficient.apply(custom_function)
#     print(result)

# if __name__ == "__main__":
#     main()












import pandas as pd
# import matplotlib as plt
import seaborn as s
import numpy as n
import matplotlib.pyplot as plt  # Correct import for matplotlib.pyplot

l= pd.read_csv('e.csv')
# print(l.head())
# print(l.describe())

# print(l.info())
# print(l.isnull().sum())
# l=l.drop('Unnamed : 0')
# print(l.head())




# l.columns = l.columns.str.strip()
# print(l.columns)


# l = l.drop('Unnamed : 0', axis=1, errors='ignore')
# l["WklyStudyHours"]=l["WklyStudyHours"].str.replace("5-oct",'5-10' )
print(l.head())



plt.figure(figsize=(4,4))
a= s.countplot(data=l,x="EthnicGroup")
counts = l['EthnicGroup'].value_counts().reset_index()
counts.columns = ['EthnicGroup', 'Count']
heatmap_data = counts.pivot_table(index='EthnicGroup', values='Count', fill_value=0)

# a.bar_label(a.containers[0])
s.heatmap(heatmap_data, annot=True, cmap='Greens', cbar=False)
plt.title('Ethnic Group Counts Heatmap')
plt.show()




