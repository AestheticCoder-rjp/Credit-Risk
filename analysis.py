# Load the data and show basic information
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('LoanHistoricalData.csv')
print("Dataset Shape (rows, columns):")
print(df.shape)
print("\
Columns available:")
print(df.columns.tolist())
print("\
Sample of the data:")
print(df.head())
print("hello")