
# import streamlit as st
# import plotly.express as px
# import pandas as pd

# def display_eda(df):
#     """
#     Perform and display exploratory data analysis.
#     """
#     st.write("## Exploratory Data Analysis")
    
#     st.write("### Dataset Shape")
#     st.write(f"Number of rows: {df.shape[0]}, Number of columns: {df.shape[1]}")

#     st.write("### Dataset Columns")
#     st.write(df.columns)

#     st.write("### Checking for Missing Values")
#     st.write(df.isnull().sum())

#     st.write("### Data Types")
#     st.write(df.dtypes)

#     st.write("### Dataset Summary")
#     st.write(df.describe())

# def display_plots(df):
#     """
#     Display various plots for the dataset using plotly.
#     """
#     st.write("## Visualizations")
    
#     # Histogram
#     st.write("### Histogram")
#     st.write("A visualization of the distribution of a numerical column's values.")
#     column = st.selectbox("Select a numerical column for histogram", df.select_dtypes(include=['number']).columns, key="histogram")
#     if column:
#         fig = px.histogram(df, x=column, nbins=20, title=f"Distribution of {column}", template='plotly_dark')
#         st.plotly_chart(fig)

#     # Box Plot
#     st.write("### Box Plot")
#     st.write("Displays the distribution of a numerical column through quartiles and outliers.")
#     box_column = st.selectbox("Select a numerical column for box plot", df.select_dtypes(include=['number']).columns, key="boxplot")
#     if box_column:
#         fig = px.box(df, y=box_column, title=f"Box Plot of {box_column}", template='plotly_dark')
#         st.plotly_chart(fig)

#     # Scatter Plot
#     st.write("### Scatter Plot")
#     st.write(" Shows the relationship between two numerical variables.")
#     scatter_x = st.selectbox("Select X-axis for scatter plot", df.select_dtypes(include=['number']).columns, key="scatter_x")
#     scatter_y = st.selectbox("Select Y-axis for scatter plot", df.select_dtypes(include=['number']).columns, key="scatter_y")
#     if scatter_x and scatter_y:
#         fig = px.scatter(df, x=scatter_x, y=scatter_y, title=f"Scatter Plot between {scatter_x} and {scatter_y}", template='plotly_dark')
#         st.plotly_chart(fig)

#     # Correlation Heatmap
#     st.write("### Correlation Heatmap")
#     st.write("A color-coded matrix showing the correlations between numerical variables.")
#     numeric_df = df.select_dtypes(include=['number'])
#     numeric_df.drop(columns=["INTEREST_RATE"], inplace=True, errors='ignore')  # Exclude specific column if needed
#     if not numeric_df.empty:
#         corr_matrix = numeric_df.corr()
#         fig = px.imshow(corr_matrix, text_auto=True, title="Correlation Heatmap", template='plotly_dark', color_continuous_scale='Blues')
#         st.plotly_chart(fig)
    
#     # Grouped Analysis
#     st.write("### Grouped Analysis")
#     st.write("A bar chart visualizing the average of a numerical column grouped by a categorical column.")
#     group_column = st.selectbox("Select a column to group by", df.columns)
#     agg_column = st.selectbox("Select a column to aggregate", df.select_dtypes(include=['int64', 'float64']).columns)
#     if group_column and agg_column:
#         grouped_data = df.groupby(group_column)[agg_column].mean().reset_index()
#         fig = px.bar(grouped_data, x=group_column, y=agg_column, title=f"Average {agg_column} by {group_column}", template='plotly_dark')
#         st.plotly_chart(fig)

import streamlit as st
import plotly.express as px
import pandas as pd

def display_eda(df):
    """
    Perform and display exploratory data analysis.
    """
    st.write("## Exploratory Data Analysis")
    
    st.write("### Dataset Shape")
    st.write(f"Number of rows: {df.shape[0]}, Number of columns: {df.shape[1]}")

    # Column Descriptions
    st.write("### Column Descriptions")
    st.write("""
    - **Loan Ids**: Unique identifiers assigned to each loan application.
    - **Good/Bad Loan**: Classification of the loan based on its repayment history.
    - **Covid_Period(Default)**: Indicates whether the loan defaulted during the COVID-19 period.
    - **RECOMENDED_LOAN_AMT**: The amount of loan that has been disbursed or recommended for disbursement.
    - **INTEREST_RATE**: The percentage of interest charged on the loan amount.
    - **TENOR**: The loan's duration or repayment period, specified in months.
    - **TOTAL_COLLATERAL_VALUE**: The total monetary value of assets pledged as security for a loan.
    - **REC_BASIS**: The criteria or rationale provided by the internal team for recommending loan approval.
    - **FRESH_TOPUP**: Indicates whether the loan is a new disbursement (fresh) or an additional amount on an existing loan (top-up).
    - **PRODUCT**: The type of loan product, categorized as secured, unsecured, or other.
    - **CONSTITUTION**: The legal entity type of the borrower, such as trust, society, private company, and others.
    - **SOURCE_BRANCH**: The originating branch responsible for processing the loan application.
    - **APPLICANT_STATE**: The state where the primary applicant associated with the loan is located.
    - **APPLICANT_CITY**: The city where the primary applicant associated with the loan is located.
    - **COBORROWER1_DESIGNATION**: The job title or role of the first co-borrower.
    - **COBORROWER2_DESIGNATION**: The job title or role of the second co-borrower.
    - **COBORROWER1_CIBILSCORE**: The credit score of the first co-borrower as provided by the CIBIL credit bureau.
    - **COBORROWER2_CIBILSCORE**: The credit score of the second co-borrower as provided by the CIBIL credit bureau.
    - **LOCALITY**: The locality or area classification of the entity or business, such as urban, rural, or semi-urban.
    - **Years of Operation**: The number of years or the operational history of the entity.
    - **TOTAL_INCOME**: The total income of the entity.
    """)


    st.write("### Checking for Missing Values")
    st.write(df.isnull().sum())

    st.write("### Data Types")
    st.write(df.dtypes)

    st.write("### Dataset Summary")
    st.write(df.describe())

def display_plots(df):
    """
    Display various plots for the dataset using plotly.
    """
    st.write("## Visualizations")
    
    # Histogram
    st.write("### Histogram")
    st.write("A visualization of the distribution of a numerical column's values.")
    column = st.selectbox("Select a numerical column for histogram", df.select_dtypes(include=['number']).columns, key="histogram")
    if column:
        fig = px.histogram(df, x=column, nbins=20, title=f"Distribution of {column}", template='plotly_dark')
        st.plotly_chart(fig)

    # Box Plot
    st.write("### Box Plot")
    st.write("Displays the distribution of a numerical column through quartiles and outliers.")
    box_column = st.selectbox("Select a numerical column for box plot", df.select_dtypes(include=['number']).columns, key="boxplot")
    if box_column:
        fig = px.box(df, y=box_column, title=f"Box Plot of {box_column}", template='plotly_dark')
        st.plotly_chart(fig)

    # Scatter Plot
    st.write("### Scatter Plot")
    st.write("Shows the relationship between two numerical variables.")
    scatter_x = st.selectbox("Select X-axis for scatter plot", df.select_dtypes(include=['number']).columns, key="scatter_x")
    scatter_y = st.selectbox("Select Y-axis for scatter plot", df.select_dtypes(include=['number']).columns, key="scatter_y")
    if scatter_x and scatter_y:
        fig = px.scatter(df, x=scatter_x, y=scatter_y, title=f"Scatter Plot between {scatter_x} and {scatter_y}", template='plotly_dark')
        st.plotly_chart(fig)

    # Correlation Heatmap
    st.write("### Correlation Heatmap")
    st.write("A color-coded matrix showing the correlations between numerical variables.")
    numeric_df = df.select_dtypes(include=['number'])
    numeric_df.drop(columns=["INTEREST_RATE"], inplace=True, errors='ignore')  # Exclude specific column if needed
    if not numeric_df.empty:
        corr_matrix = numeric_df.corr()
        fig = px.imshow(corr_matrix, text_auto=True, title="Correlation Heatmap", template='plotly_dark', color_continuous_scale='Blues')
        st.plotly_chart(fig)
    
    # Grouped Analysis
    st.write("### Grouped Analysis")
    st.write("A bar chart visualizing the average of a numerical column grouped by a categorical column.")
    group_column = st.selectbox("Select a column to group by", df.columns)
    agg_column = st.selectbox("Select a column to aggregate", df.select_dtypes(include=['int64', 'float64']).columns)
    if group_column and agg_column:
        grouped_data = df.groupby(group_column)[agg_column].mean().reset_index()
        fig = px.bar(grouped_data, x=group_column, y=agg_column, title=f"Average {agg_column} by {group_column}", template='plotly_dark')
        st.plotly_chart(fig)
