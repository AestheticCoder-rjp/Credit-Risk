import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu

# Import the 'data' function from the 'analysis' module
from analysis import data

selected = option_menu( 
                menu_title=None, 
                options = ["EDA", "Analysis","Model"],
                icons= ["activity","bar-chart","brilliance"] ,menu_icon="cast",
                        default_index=0,
                        orientation="horizontal" )

# Call the 'data' function and store its result
df = data()

# Streamlit app
st.title("Loan Historical Data Analysis")

if isinstance(df, pd.DataFrame):  # Ensure df is a valid DataFrame
    st.write("### Dataset Shape")
    st.write(df.shape)
    
    st.write("### Dataset Columns")
    st.write(df.columns)

    st.write("### Checking Null Values")
    st.write(df.isnull().sum())
    
    st.write("### Checking Data Types")
    st.write(df.dtypes)
    
    st.write("### Dataset Summary")
    st.write(df.describe())
    
    # Plots Section
    st.write("## Data Visualization")
    
    # Select column for histogram
    st.write("### Histogram")
    column = st.selectbox("Select a column for histogram", df.select_dtypes(include=['int64', 'float64']).columns)
    if column:
        fig, ax = plt.subplots()
        sns.histplot(df[column], kde=True, ax=ax)
        st.pyplot(fig)
    
    # Box Plot
    st.write("### Box Plot")
    box_column = st.selectbox("Select a column for box plot", df.select_dtypes(include=['int64', 'float64']).columns)
    if box_column:
        fig, ax = plt.subplots()
        sns.boxplot(data=df, y=box_column, ax=ax)
        st.pyplot(fig)
    
    # Scatter Plot
    st.write("### Scatter Plot")
    scatter_x = st.selectbox("Select X-axis for scatter plot", df.select_dtypes(include=['int64', 'float64']).columns)
    scatter_y = st.selectbox("Select Y-axis for scatter plot", df.select_dtypes(include=['int64', 'float64']).columns)
    if scatter_x and scatter_y:
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x=scatter_x, y=scatter_y, ax=ax)
        st.pyplot(fig)
    
        
    # Correlation Heatmap
    st.write("### Correlation Heatmap")
    try:
        numeric_df = df.select_dtypes(include=['number'])  # Select only numeric columns
        numeric_df.drop("INTEREST_RATE",axis=1,inplace=True, errors="ignore")
        if numeric_df.empty:
            st.write("No numeric columns available for correlation heatmap.")
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
    except Exception as e:
        st.write(f"Error in generating correlation heatmap: {e}")

    
    # Grouped Analysis
    st.write("### Grouped Analysis")
    group_column = st.selectbox("Select a column to group by", df.columns)
    agg_column = st.selectbox("Select a column to aggregate", df.select_dtypes(include=['int64', 'float64']).columns)
    if group_column and agg_column:
        grouped_data = df.groupby(group_column)[agg_column].mean()
        st.write(grouped_data)
else:
    st.write(df)



#     # Display selected section
#     if selected == "EDA":
#         display_eda(data)
#     elif selected == "Plots":
#         display_plots(data)
#     elif selected == "Predictions":
#         make_predictions(data)


# if __name__ == "__main__":
#     main()
