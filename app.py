import streamlit as st

# Import the 'data' function from the 'analysis' module
from analysis import data

# Call the 'data' function and store its result
result = data()

# Streamlit app
st.title("Loan Historical Data Analysis")

if isinstance(result, dict):
    # Display dataset information
    st.write("### Dataset Shape")
    st.write(result["Dataset Shape"])
    st.write("### Columns")
    st.write(result["Columns"])
    st.write("### Sample Data")
    st.dataframe(result["Sample Data"]) 
    # Use Streamlit's dataframe display
    st.write(result['Data Types'])
elif isinstance(result, str):
    # Display error message
    st.error(result)
else:
    st.write("Unexpected result from the 'data' function.")
