import pandas as pd

def data():
    try:
        # Load the CSV file
        df = pd.read_csv('LoanHistoricalData.csv')
        
        # Return basic information as a dictionary for Streamlit display
        return df
    except FileNotFoundError:
        return "File 'LoanHistoricalData.csv' not found. Please check the file path."
    except Exception as e:
        return f"An error occurred: {e}"
