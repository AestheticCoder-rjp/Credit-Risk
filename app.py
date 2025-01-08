import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from eda import display_eda, display_plots
from analysis import generate_insights
from model import loan_status_prediction_app

def main():
    st.markdown("# Loan Historical Data Analysis")
    
    # Load the dataset
    try:
        df = pd.read_csv("LoanHistoricalData.csv")
    except FileNotFoundError:
        st.error("Dataset not found. Please check the file path.")
        return
    
    # Sidebar navigation
    selected = option_menu(
        menu_title=None, 
        options=["EDA", "Analysis", "Modeling"],
        icons=["activity", "bar-chart", "brilliance"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal"
    )
    
    # Display selected section
    if selected == "EDA":
        display_eda(df)
        display_plots(df)
    elif selected == "Analysis":
        st.write("## Top Insights")
        
        # Generate insights
        insights = generate_insights(df)

        # Sidebar navigation for insights
        options = list(insights.keys())
        choice = st.sidebar.selectbox("Select an analysis to display", options)

        # Display selected insight
        if choice in insights:
            st.plotly_chart(insights[choice])
    elif selected == "Modeling":
        # st.write("## Modeling Section (Work in Progress)")
        # a,b=preprocess_data('LoanHistoricalData.csv')
        # st.write("## Model Performance Comparison")
        # results_df = train_and_evaluate_models(a)
        # st.write(results_df)
        # st.pyplot(visualize_results(results_df))
        # Run the app
        #loan_status_prediction_app(df)

# Now call the Streamlit app function
        loan_status_prediction_app(df)


        
if __name__ == "__main__":
    main()
