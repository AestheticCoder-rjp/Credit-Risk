# import streamlit as st
# import seaborn as sns
# import matplotlib.pyplot as plt

# def display_eda(df):
#     """
#     Perform and display exploratory data analysis.
#     """
#     st.write("## Exploratory Data Analysis")
    
#     st.write("### Dataset Shape")
#     st.write(df.shape)

#     st.write("### Dataset Columns")
#     st.write(df.columns)

#     st.write("### Checking Null Values")
#     st.write(df.isnull().sum())

#     st.write("### Checking Data Types")
#     st.write(df.dtypes)

#     st.write("### Dataset Summary")
#     st.write(df.describe())

# def display_plots(df):
#     """
#     Display various plots for the dataset.
#     """
#     st.write("## Visualizations")
    
#     # Histogram
#     st.write("### Histogram")
#     column = st.selectbox("Select a column for histogram", df.select_dtypes(include=['number']).columns, key="histogram")
#     if column:
#         fig, ax = plt.subplots()
#         sns.histplot(df[column], kde=True, ax=ax)
#         st.pyplot(fig)

#     # Box Plot
#     st.write("### Box Plot")
#     box_column = st.selectbox("Select a column for box plot", df.select_dtypes(include=['number']).columns, key="boxplot")
#     if box_column:
#         fig, ax = plt.subplots()
#         sns.boxplot(data=df, y=box_column, ax=ax)
#         st.pyplot(fig)

#     # Scatter Plot
#     st.write("### Scatter Plot")
#     scatter_x = st.selectbox("Select X-axis for scatter plot", df.select_dtypes(include=['number']).columns, key="scatter_x")
#     scatter_y = st.selectbox("Select Y-axis for scatter plot", df.select_dtypes(include=['number']).columns, key="scatter_y")
#     if scatter_x and scatter_y:
#         fig, ax = plt.subplots()
#         sns.scatterplot(data=df, x=scatter_x, y=scatter_y, ax=ax)
#         st.pyplot(fig)

#     # Correlation Heatmap
#     st.write("### Correlation Heatmap")
#     numeric_df = df.select_dtypes(include=['number'])
#     numeric_df.drop(columns=["INTEREST_RATE"], inplace=True)
#     if not numeric_df.empty:
#         fig, ax = plt.subplots(figsize=(10, 6))
#         sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
#         st.pyplot(fig)
    
#     st.write("### Grouped Analysis")
#     group_column = st.selectbox("Select a column to group by", df.columns)
#     agg_column = st.selectbox("Select a column to aggregate", df.select_dtypes(include=['int64', 'float64']).columns)
#     if group_column and agg_column:
#         grouped_data = df.groupby(group_column)[agg_column].mean()
#         st.write(grouped_data)

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

    st.write("### Dataset Columns")
    st.write(df.columns)

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
    column = st.selectbox("Select a numerical column for histogram", df.select_dtypes(include=['number']).columns, key="histogram")
    if column:
        fig = px.histogram(df, x=column, nbins=20, title=f"Distribution of {column}", template='plotly_dark')
        st.plotly_chart(fig)

    # Box Plot
    st.write("### Box Plot")
    box_column = st.selectbox("Select a numerical column for box plot", df.select_dtypes(include=['number']).columns, key="boxplot")
    if box_column:
        fig = px.box(df, y=box_column, title=f"Box Plot of {box_column}", template='plotly_dark')
        st.plotly_chart(fig)

    # Scatter Plot
    st.write("### Scatter Plot")
    scatter_x = st.selectbox("Select X-axis for scatter plot", df.select_dtypes(include=['number']).columns, key="scatter_x")
    scatter_y = st.selectbox("Select Y-axis for scatter plot", df.select_dtypes(include=['number']).columns, key="scatter_y")
    if scatter_x and scatter_y:
        fig = px.scatter(df, x=scatter_x, y=scatter_y, title=f"Scatter Plot between {scatter_x} and {scatter_y}", template='plotly_dark')
        st.plotly_chart(fig)

    # Correlation Heatmap
    st.write("### Correlation Heatmap")
    numeric_df = df.select_dtypes(include=['number'])
    numeric_df.drop(columns=["INTEREST_RATE"], inplace=True, errors='ignore')  # Exclude specific column if needed
    if not numeric_df.empty:
        corr_matrix = numeric_df.corr()
        fig = px.imshow(corr_matrix, text_auto=True, title="Correlation Heatmap", template='plotly_dark', color_continuous_scale='Blues')
        st.plotly_chart(fig)
    
    # Grouped Analysis
    st.write("### Grouped Analysis")
    group_column = st.selectbox("Select a column to group by", df.columns)
    agg_column = st.selectbox("Select a column to aggregate", df.select_dtypes(include=['int64', 'float64']).columns)
    if group_column and agg_column:
        grouped_data = df.groupby(group_column)[agg_column].mean().reset_index()
        fig = px.bar(grouped_data, x=group_column, y=agg_column, title=f"Average {agg_column} by {group_column}", template='plotly_dark')
        st.plotly_chart(fig)
