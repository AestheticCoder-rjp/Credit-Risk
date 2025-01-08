# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from xgboost import XGBClassifier
# from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
# import matplotlib.pyplot as plt

# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import LabelEncoder

# def preprocess_data(file_path):
#     """
#     Preprocess the data by handling missing values, encoding, and scaling.
#     """
#     df = pd.read_csv(file_path)

#     # Separate columns by type
#     numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
#     categorical_cols = df.select_dtypes(include=['object']).columns

#     # Impute numeric columns
#     imputer_num = SimpleImputer(strategy='mean')
#     df[numeric_cols] = imputer_num.fit_transform(df[numeric_cols])

#     # Impute categorical columns
#     imputer_cat = SimpleImputer(strategy='most_frequent')
#     df[categorical_cols] = imputer_cat.fit_transform(df[categorical_cols])

#     # Encode categorical variables
#     label_encoders = {}
#     for col in categorical_cols:
#         le = LabelEncoder()
#         df[col] = le.fit_transform(df[col].astype(str))
#         label_encoders[col] = le

#     # Final check for NaNs
#     if df.isnull().sum().any():
#         raise ValueError("Imputation failed; NaNs remain in the dataset.")

#     return df, label_encoders




# def train_and_evaluate_models(df):
#     """
#     Train and evaluate multiple machine learning models and return results.
#     """
#     y = df['Good/Bad Loan']
#     X = df.drop(columns=['Good/Bad Loan', 'Loan Ids','Covid_Period(Default)','INTEREST_RATE','REC_BASIS','FRESH_TOPUP','PRODUCT','CONSTITUTION','SOURCE_BRANCH','APPLICANT_STATE','APPLICANT_CITY','COBORROWER1_DESIGNATION','COBORROWER2_DESIGNATION','LOCALITY','COBORROWER1_CIBILSCORE','COBORROWER2_CIBILSCORE','TOTAL_COLLATERAL_VALUE'], errors='ignore')

#     # Split the data
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Scale the data
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)

#     # Models to evaluate
#     models = {
#         'Logistic Regression': LogisticRegression(),
#         'Random Forest': RandomForestClassifier(),
#         'Gradient Boosting': GradientBoostingClassifier(),
#         'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
#     }

#     results = {}
#     for model_name, model in models.items():
#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_test)
#         y_prob = model.predict_proba(X_test)[:, 1]

#         # Metrics
#         accuracy = accuracy_score(y_test, y_pred)
#         roc_auc = roc_auc_score(y_test, y_prob)
#         report = classification_report(y_test, y_pred, output_dict=True)

#         results[model_name] = {
#             'Accuracy': accuracy,
#             'ROC AUC': roc_auc,
#             'Precision': report['1']['precision'],
#             'Recall': report['1']['recall'],
#             'F1-Score': report['1']['f1-score']
#         }

#     return pd.DataFrame(results).T

# def visualize_results(results_df):
#     """
#     Visualize the model performance metrics.
#     """
#     results_df[['Accuracy', 'ROC AUC', 'Precision', 'Recall', 'F1-Score']].plot(kind='bar', figsize=(12, 8))
#     plt.title('Model Performance Comparison')
#     plt.ylabel('Score')
#     plt.xlabel('Model')
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.show()


# import streamlit as st
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.impute import SimpleImputer
# from sklearn.linear_model import LogisticRegression

# def loan_status_prediction_app(df):
#     """
#     Streamlit app for loan default prediction using logistic regression.
    
#     Parameters:
#     - df (pd.DataFrame): DataFrame containing pre-trained model data and configurations.
#     """
#     # Streamlit app title
#     st.title("Loan Default Prediction")

#     # Description
#     st.write("""
#     This app predicts whether a loan is Good or Bad based on user inputs for key features. 
#     Please provide the required values below:
#     """)

#     # Columns for the model (must match the trained model's feature set)
#     columns = [
#         'COBORROWER1_CIBILSCORE', 'COBORROWER2_CIBILSCORE', 'TOTAL_INCOME',
#         'RECOMENDED_LOAN_AMT', 'Covid_Period(Default)',
#         'Years of Operation', 'PRODUCT', 'TOTAL_COLLATERAL_VALUE'
#     ]

#     # Input fields for user data
#     input_data = {}
#     for column in columns:
#         if column == 'PRODUCT' or column == 'Covid_Period(Default)':
#             input_data[column] = st.selectbox(f"Enter {column} (0 or 1):", [0, 1])
#         else:
#             input_data[column] = st.number_input(f"Enter {column}:", min_value=0.0, value=0.0)

#     # Button to predict
#     if st.button("Predict Loan Status"):
#         # Ensure df has the required columns and preprocessing
#         columns_to_use = [
#             'COBORROWER1_CIBILSCORE', 'COBORROWER2_CIBILSCORE', 'TOTAL_INCOME',
#             'RECOMENDED_LOAN_AMT', 'Covid_Period(Default)',
#             'Years of Operation', 'PRODUCT', 'TOTAL_COLLATERAL_VALUE', 'Good/Bad Loan'
#         ]
#         df = df[columns_to_use]

#         # Encoding categorical columns using LabelEncoder
#         label_encoder_product = LabelEncoder()
#         df['PRODUCT'] = label_encoder_product.fit_transform(df['PRODUCT'])
        
#         label_encoder_covid = LabelEncoder()
#         df['Covid_Period(Default)'] = label_encoder_covid.fit_transform(df['Covid_Period(Default)'])
        
#         df['Good/Bad Loan'] = label_encoder_product.fit_transform(df['Good/Bad Loan'])

#         # Handling missing values (imputation)
#         imputer = SimpleImputer(strategy='median')
#         df.iloc[:, :-1] = imputer.fit_transform(df.iloc[:, :-1])

#         # Splitting the data into train/test
#         X = df.drop(columns=['Good/Bad Loan'])
#         y = df['Good/Bad Loan']
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#         # Scaling the data
#         scaler = StandardScaler()
#         X_train = scaler.fit_transform(X_train)
#         X_test = scaler.transform(X_test)

#         # Logistic regression model
#         logistic_model = LogisticRegression(
#             C=0.1,
#             penalty='l1',
#             solver='liblinear',
#             random_state=42
#         )

#         # Train the model
#         logistic_model.fit(X_train, y_train)

#         # Now process the user input data
#         input_df = pd.DataFrame([input_data])

#         # Ensure input_df has all columns in the same order as during training
#         input_df = input_df[columns]

#         # Scale the user input data
#         input_df[['COBORROWER1_CIBILSCORE', 'COBORROWER2_CIBILSCORE', 'TOTAL_INCOME',
#                   'RECOMENDED_LOAN_AMT', 'Covid_Period(Default)','Years of Operation','PRODUCT', 'TOTAL_COLLATERAL_VALUE']] = \
#             scaler.transform(input_df[['COBORROWER1_CIBILSCORE', 'COBORROWER2_CIBILSCORE', 'TOTAL_INCOME',
#                                        'RECOMENDED_LOAN_AMT','Covid_Period(Default)', 'Years of Operation','PRODUCT', 'TOTAL_COLLATERAL_VALUE']])

#         # Apply label encoding to categorical columns using the same encoders used for training data
#         #input_df['PRODUCT'] = label_encoder_product.transform(input_df['PRODUCT'])
#         #input_df['Covid_Period(Default)'] = label_encoder_covid.transform(input_df['Covid_Period(Default)'])

#         # Make the prediction
#         prediction = logistic_model.predict(input_df)[0]
#         prediction_text = "Good Loan" if prediction == 1 else "Bad Loan"

#         # Display the prediction
#         st.write(f"### Prediction: {prediction_text}")
#     else:
#         st.write("Provide inputs and click 'Predict Loan Status' to see the prediction.")

# # Example usage:
# # Load pre-trained model data
# sample_data = pd.DataFrame({
#     'COBORROWER1_CIBILSCORE': np.random.randint(300, 900, size=100),
#     'COBORROWER2_CIBILSCORE': np.random.randint(300, 900, size=100),
#     'TOTAL_INCOME': np.random.randint(10000, 100000, size=100),
#     'RECOMENDED_LOAN_AMT': np.random.randint(5000, 50000, size=100),
#     'Covid_Period(Default)': np.random.randint(0, 2, size=100),
#     'Years of Operation': np.random.randint(1, 30, size=100),
#     'PRODUCT': np.random.randint(0, 2, size=100),
#     'TOTAL_COLLATERAL_VALUE': np.random.randint(1000, 50000, size=100),
#     'Good/Bad Loan': np.random.randint(0, 2, size=100),
# })

# import streamlit as st
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.impute import SimpleImputer
# from sklearn.linear_model import LogisticRegression

# def loan_status_prediction_app(df):
#     """
#     Streamlit app for loan default prediction using logistic regression.
    
#     Parameters:
#     - df (pd.DataFrame): DataFrame containing pre-trained model data and configurations.
#     """
#     # Streamlit app title
#     st.title("Loan Default Prediction")

#     # Description
#     st.write("""
#     This app predicts whether a loan is likely to be approved or rejected based on various financial details.
#     Please fill in the details below to get the prediction:
#     """)

#     # Input field explanations (simplified for user-friendliness)
#     input_explanations = {
#         'COBORROWER1_CIBILSCORE': 'CIBIL score of the first co-borrower (higher is better, between 300-900)',
#         'COBORROWER2_CIBILSCORE': 'CIBIL score of the second co-borrower (higher is better, between 300-900)',
#         'TOTAL_INCOME': 'Total annual income of the borrower (in currency units)',
#         'RECOMENDED_LOAN_AMT': 'Recommended loan amount by the system (in currency units)',
#         'Covid_Period(Default)': 'Was the loan affected by the Covid-19 period? (Yes = 1, No = 0)',
#         'Years of Operation': 'How many years has the borrower been in business?',
#         'PRODUCT': 'Type of loan product (Specific product = 1, Others = 0)',
#         'TOTAL_COLLATERAL_VALUE': 'Total value of collateral available for the loan (in currency units)'
#     }

#     # Columns for the model (must match the trained model's feature set)
#     columns = [
#         'COBORROWER1_CIBILSCORE', 'COBORROWER2_CIBILSCORE', 'TOTAL_INCOME',
#         'RECOMENDED_LOAN_AMT', 'Covid_Period(Default)',
#         'Years of Operation', 'PRODUCT', 'TOTAL_COLLATERAL_VALUE'
#     ]

#     # Input fields for user data
#     input_data = {}
#     for column in columns:
#         # Display explanation for each column
#         st.write(f"**{column.replace('_', ' ')}**: {input_explanations[column]}")
        
#         # Collect input from the user
#         if column == 'PRODUCT' or column == 'Covid_Period(Default)':
#             input_data[column] = st.selectbox(f"Select {column.replace('_', ' ')} (0 or 1):", [0, 1])
#         else:
#             input_data[column] = st.number_input(f"Enter {column.replace('_', ' ')}:", min_value=0.0, value=0.0)

#     # Button to predict
#     if st.button("Predict Loan Status"):
#         # Ensure df has the required columns and preprocessing
#         columns_to_use = [
#             'COBORROWER1_CIBILSCORE', 'COBORROWER2_CIBILSCORE', 'TOTAL_INCOME',
#             'RECOMENDED_LOAN_AMT', 'Covid_Period(Default)',
#             'Years of Operation', 'PRODUCT', 'TOTAL_COLLATERAL_VALUE', 'Good/Bad Loan'
#         ]
#         df = df[columns_to_use]

#         # Encoding categorical columns using LabelEncoder
#         label_encoder_product = LabelEncoder()
#         df['PRODUCT'] = label_encoder_product.fit_transform(df['PRODUCT'])
        
#         label_encoder_covid = LabelEncoder()
#         df['Covid_Period(Default)'] = label_encoder_covid.fit_transform(df['Covid_Period(Default)'])
        
#         df['Good/Bad Loan'] = label_encoder_product.fit_transform(df['Good/Bad Loan'])

#         # Handling missing values (imputation)
#         imputer = SimpleImputer(strategy='median')
#         df.iloc[:, :-1] = imputer.fit_transform(df.iloc[:, :-1])

#         # Splitting the data into train/test
#         X = df.drop(columns=['Good/Bad Loan'])
#         y = df['Good/Bad Loan']
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#         # Scaling the data
#         scaler = StandardScaler()
#         X_train = scaler.fit_transform(X_train)
#         X_test = scaler.transform(X_test)

#         # Logistic regression model
#         logistic_model = LogisticRegression(
#             C=0.1,
#             penalty='l1',
#             solver='liblinear',
#             random_state=42
#         )

#         # Train the model
#         logistic_model.fit(X_train, y_train)

#         # Now process the user input data
#         input_df = pd.DataFrame([input_data])

#         # Ensure input_df has all columns in the same order as during training
#         input_df = input_df[columns]

#         # Scale the user input data
#         input_df[['COBORROWER1_CIBILSCORE', 'COBORROWER2_CIBILSCORE', 'TOTAL_INCOME',
#                   'RECOMENDED_LOAN_AMT', 'Covid_Period(Default)','Years of Operation','PRODUCT', 'TOTAL_COLLATERAL_VALUE']] = \
#             scaler.transform(input_df[['COBORROWER1_CIBILSCORE', 'COBORROWER2_CIBILSCORE', 'TOTAL_INCOME',
#                                        'RECOMENDED_LOAN_AMT','Covid_Period(Default)', 'Years of Operation','PRODUCT', 'TOTAL_COLLATERAL_VALUE']])

#         # Make the prediction
#         prediction = logistic_model.predict(input_df)[0]
#         prediction_text = "Good Loan" if prediction == 1 else "Bad Loan"

#         # Beautify the prediction display
#         if prediction == 1:
#             st.markdown(f"<h3 style='color: green; font-weight: bold;'>🎉 {prediction_text}</h3>", unsafe_allow_html=True)
#         else:
#             st.markdown(f"<h3 style='color: red; font-weight: bold;'>❌ {prediction_text}</h3>", unsafe_allow_html=True)
#     else:
#         st.write("Provide inputs and click 'Predict Loan Status' to see the prediction.")

# # Example usage:
# # Load pre-trained model data
# sample_data = pd.DataFrame({
#     'COBORROWER1_CIBILSCORE': np.random.randint(300, 900, size=100),
#     'COBORROWER2_CIBILSCORE': np.random.randint(300, 900, size=100),
#     'TOTAL_INCOME': np.random.randint(10000, 100000, size=100),
#     'RECOMENDED_LOAN_AMT': np.random.randint(5000, 50000, size=100),
#     'Covid_Period(Default)': np.random.randint(0, 2, size=100),
#     'Years of Operation': np.random.randint(1, 30, size=100),
#     'PRODUCT': np.random.randint(0, 2, size=100),
#     'TOTAL_COLLATERAL_VALUE': np.random.randint(1000, 50000, size=100),
#     'Good/Bad Loan': np.random.randint(0, 2, size=100),
# })

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
    st.title("Loan Default Prediction")

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
