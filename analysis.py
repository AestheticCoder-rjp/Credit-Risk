
import pandas as pd
import plotly.express as px
import streamlit as st

def generate_insights(df):
    """
    Generate all insights and plots from the dataset.

    Parameters:
        df (DataFrame): The loan historical data.
    """
    
    # Define descriptions for each plot
    plot_descriptions = {
        'Default by Product': "This plot shows the default rates for each loan product type. It compares the percentage of good and bad loans for different product types.Secured Loans have the highest default rate.",
        'Collateral Distribution': "This box plot visualizes the distribution of collateral values based on loan performance. It shows the spread and outliers for good and bad loans.",
        'Default by State': "This plot compares the default rates across different states. It shows the percentage of good and bad loans for each state.Ohio State has the highest default rate.",
        'Income Distribution': "This box plot visualizes the income distribution of borrowers based on loan performance. It compares the total income for good and bad loans.",
        'COVID Default Rates': "This plot compares the default rates between the COVID period and non-COVID periods. It shows how the default rates have changed during the pandemic.During the COVID period, the default rate is higher."
    }

    # Sidebar for selecting the graph to display
    graph_options = ['Default by Product', 'Collateral Distribution', 'Default by State', 'Income Distribution', 'COVID Default Rates']
    selected_graph = st.sidebar.selectbox("Select a Graph", graph_options)

    # Show the description of the selected graph
    st.write(f"### Description of {selected_graph}")
    st.write(plot_descriptions[selected_graph])

    # Generate and display the selected graph
    if selected_graph == 'Default by Product':
        default_by_product = df.groupby('PRODUCT')['Good/Bad Loan'].value_counts(normalize=True).unstack() * 100
        fig = px.bar(
            default_by_product,
            title='Default Rates by Product Type',
            barmode='group',
            labels={'value': 'Percentage', 'PRODUCT': 'Product Type'},
            color_discrete_sequence=['#EF553B', '#636EFA']
        ).update_layout(yaxis_title='Percentage', xaxis_title='Product Type')
        st.plotly_chart(fig, key="default_by_product")

    elif selected_graph == 'Collateral Distribution':
        fig = px.box(
            df,
            x='Good/Bad Loan',
            y='TOTAL_COLLATERAL_VALUE',
            title='Collateral Value Distribution by Loan Performance',
            color='Good/Bad Loan',
            labels={'TOTAL_COLLATERAL_VALUE': 'Collateral Value', 'Good/Bad Loan': 'Loan Performance'}
        )
        st.plotly_chart(fig, key="collateral_distribution")

    elif selected_graph == 'Default by State':
        default_by_state = df.groupby('APPLICANT_STATE')['Good/Bad Loan'].value_counts(normalize=True).unstack() * 100
        fig = px.bar(
            default_by_state,
            title='Default Rates by State',
            barmode='group',
            labels={'value': 'Percentage', 'APPLICANT_STATE': 'State'},
            color_discrete_sequence=['#EF553B', '#636EFA']
        ).update_layout(yaxis_title='Percentage', xaxis_title='State')
        st.plotly_chart(fig, key="default_by_state")

    elif selected_graph == 'Income Distribution':
        fig = px.box(
            df,
            x='Good/Bad Loan',
            y='TOTAL_INCOME',
            title='Income Distribution by Loan Performance',
            color='Good/Bad Loan',
            labels={'TOTAL_INCOME': 'Total Income', 'Good/Bad Loan': 'Loan Performance'}
        )
        st.plotly_chart(fig, key="income_distribution")

    elif selected_graph == 'COVID Default Rates':
        covid_default = df.groupby('Covid_Period(Default)')['Good/Bad Loan'].value_counts(normalize=True).unstack() * 100
        fig = px.bar(
            covid_default,
            title='Default Rates During COVID vs Non-COVID Periods',
            barmode='group',
            labels={'value': 'Percentage', 'Covid_Period(Default)': 'COVID Period'},
            color_discrete_sequence=['#EF553B', '#636EFA']
        ).update_layout(yaxis_title='Percentage', xaxis_title='COVID Period')
        st.plotly_chart(fig, key="covid_default_rates")
