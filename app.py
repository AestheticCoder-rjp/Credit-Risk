import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Import the 'data' function or variable from the 'analysis' module
from analysis import data

# Call the 'data' function and store its result
result = data()
st.write(result)
# Streamlit app
st.write("Hello, Streamlit!")
# if result is not None:
#     st.write(result)
# else:
#     st.write("The 'data' function returned None or is not implemented correctly.")

df=pd.read_csv('D:\@CU\Case_Study\Credit-Risk\LoanHistoricalData.csv')
st.write(df.head())