import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

def loan_status_prediction_app(df):
    """
    Streamlit app for loan default prediction using logistic regression.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing pre-trained model data and configurations.
    """
    # Streamlit app title
    st.header("Loan Default Prediction",divider="rainbow")

    # Description
    st.write("""
    This app predicts whether a loan is likely to be approved or rejected based on various financial details.
    Please fill in the details below to get the prediction:
    """)

    # Input field explanations (simplified for user-friendliness)
    input_explanations = {
        'COBORROWER1_CIBILSCORE': 'CIBIL score of the first co-borrower (higher is better, between 300-900)',
        'COBORROWER2_CIBILSCORE': 'CIBIL score of the second co-borrower (higher is better, between 300-900)',
        'TOTAL_INCOME': 'Total annual income of the borrower (in currency units)',
        'RECOMENDED_LOAN_AMT': 'Recommended loan amount by the system (in currency units)',
        'Covid_Period(Default)': 'Was the loan affected by the Covid-19 period? (Yes = 1, No = 0)',
        'Years of Operation': 'How many years has the borrower been in business?',
        'PRODUCT': 'Type of loan product (Secured = 1, Others = 0)',
        'TOTAL_COLLATERAL_VALUE': 'Total value of collateral available for the loan (in currency units)'
    }

    # Columns for the model (must match the trained model's feature set)
    columns = [
        'COBORROWER1_CIBILSCORE', 'COBORROWER2_CIBILSCORE', 'TOTAL_INCOME',
        'RECOMENDED_LOAN_AMT', 'Covid_Period(Default)',
        'Years of Operation', 'PRODUCT', 'TOTAL_COLLATERAL_VALUE'
    ]

    # Default values
    default_values = {
        'COBORROWER1_CIBILSCORE': 866.00,
        'COBORROWER2_CIBILSCORE': 862.00,
        'TOTAL_INCOME': 2708374.00,
        'RECOMENDED_LOAN_AMT': 77285.00,
        'Covid_Period(Default)': 0,  # No (Covid)
        'Years of Operation': 29.00,
        'PRODUCT': 1,  # Secured
        'TOTAL_COLLATERAL_VALUE': 50000.00  # Example value, can be adjusted
    }

    # Input fields for user data with default values
    input_data = {}
    for column in columns:
        # Display explanation for each column
        st.write(f"**{column.replace('_', ' ')}**: {input_explanations[column]}")
        
        # Collect input from the user with default values
        if column == 'PRODUCT' or column == 'Covid_Period(Default)':
            input_data[column] = st.selectbox(f"Select {column.replace('_', ' ')} (0 or 1):", [0, 1], index=default_values[column])
        else:
            input_data[column] = st.number_input(f"Enter {column.replace('_', ' ')}:", min_value=0.0, value=default_values[column])

    # Button to predict
    if st.button("Predict Loan Status"):
        # Ensure df has the required columns and preprocessing
        columns_to_use = [
            'COBORROWER1_CIBILSCORE', 'COBORROWER2_CIBILSCORE', 'TOTAL_INCOME',
            'RECOMENDED_LOAN_AMT', 'Covid_Period(Default)',
            'Years of Operation', 'PRODUCT', 'TOTAL_COLLATERAL_VALUE', 'Good/Bad Loan'
        ]
        df = df[columns_to_use]

        # Encoding categorical columns using LabelEncoder
        label_encoder_product = LabelEncoder()
        df['PRODUCT'] = label_encoder_product.fit_transform(df['PRODUCT'])
        
        label_encoder_covid = LabelEncoder()
        df['Covid_Period(Default)'] = label_encoder_covid.fit_transform(df['Covid_Period(Default)'])
        
        df['Good/Bad Loan'] = label_encoder_product.fit_transform(df['Good/Bad Loan'])

        # Handling missing values (imputation)
        imputer = SimpleImputer(strategy='median')
        df.iloc[:, :-1] = imputer.fit_transform(df.iloc[:, :-1])

        # Splitting the data into train/test
        X = df.drop(columns=['Good/Bad Loan'])
        y = df['Good/Bad Loan']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scaling the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Logistic regression model
        logistic_model = LogisticRegression(
            C=0.1,
            penalty='l1',
            solver='liblinear',
            random_state=42
        )

        # Train the model
        logistic_model.fit(X_train, y_train)

        # Now process the user input data
        input_df = pd.DataFrame([input_data])

        # Ensure input_df has all columns in the same order as during training
        input_df = input_df[columns]

        # Scale the user input data
        input_df[['COBORROWER1_CIBILSCORE', 'COBORROWER2_CIBILSCORE', 'TOTAL_INCOME',
                  'RECOMENDED_LOAN_AMT', 'Covid_Period(Default)','Years of Operation','PRODUCT', 'TOTAL_COLLATERAL_VALUE']] = \
            scaler.transform(input_df[['COBORROWER1_CIBILSCORE', 'COBORROWER2_CIBILSCORE', 'TOTAL_INCOME',
                                       'RECOMENDED_LOAN_AMT','Covid_Period(Default)', 'Years of Operation','PRODUCT', 'TOTAL_COLLATERAL_VALUE']])

        # Make the prediction
        prediction = logistic_model.predict(input_df)[0]
        prediction_text = "Good Loan" if prediction == 1 else "Bad Loan"

        # Beautify the prediction display
        if prediction == 1:
            st.markdown(f"<h3 style='color: green; font-weight: bold;'>🎉 {prediction_text}</h3>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h3 style='color: red; font-weight: bold;'>❌ {prediction_text}</h3>", unsafe_allow_html=True)
    else:
        st.write("Provide inputs and click 'Predict Loan Status' to see the prediction.")

# Example usage:
# Load pre-trained model data
sample_data = pd.DataFrame({
    'COBORROWER1_CIBILSCORE': np.random.randint(300, 900, size=100),
    'COBORROWER2_CIBILSCORE': np.random.randint(300, 900, size=100),
    'TOTAL_INCOME': np.random.randint(10000, 100000, size=100),
    'RECOMENDED_LOAN_AMT': np.random.randint(5000, 50000, size=100),
    'Covid_Period(Default)': np.random.randint(0, 2, size=100),
    'Years of Operation': np.random.randint(1, 30, size=100),
    'PRODUCT': np.random.randint(0, 2, size=100),
    'TOTAL_COLLATERAL_VALUE': np.random.randint(1000, 50000, size=100),
    'Good/Bad Loan': np.random.randint(0, 2, size=100),
})
