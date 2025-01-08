import pandas as pd
import plotly.express as px

def generate_insights(df):
    """
    Generate all insights and plots from the dataset.

    Parameters:
        df (DataFrame): The loan historical data.

    Returns:
        dict: A dictionary of plotly figures for each insight.
    """
    insights = {}

    # Insight 1: Default rates by product type
    default_by_product = df.groupby('PRODUCT')['Good/Bad Loan'].value_counts(normalize=True).unstack() * 100
    insights['Default by Product'] = px.bar(
        default_by_product,
        title='Default Rates by Product Type',
        barmode='group',
        labels={'value': 'Percentage', 'PRODUCT': 'Product Type'},
        color_discrete_sequence=['#EF553B', '#636EFA']
    ).update_layout(yaxis_title='Percentage', xaxis_title='Product Type')

    # Insight 2: Collateral value distribution by loan performance
    insights['Collateral Distribution'] = px.box(
        df,
        x='Good/Bad Loan',
        y='TOTAL_COLLATERAL_VALUE',
        title='Collateral Value Distribution by Loan Performance',
        color='Good/Bad Loan',
        labels={'TOTAL_COLLATERAL_VALUE': 'Collateral Value', 'Good/Bad Loan': 'Loan Performance'}
    )

    # Insight 3: Default rates by state
    default_by_state = df.groupby('APPLICANT_STATE')['Good/Bad Loan'].value_counts(normalize=True).unstack() * 100
    insights['Default by State'] = px.bar(
        default_by_state,
        title='Default Rates by State',
        barmode='group',
        labels={'value': 'Percentage', 'APPLICANT_STATE': 'State'},
        color_discrete_sequence=['#EF553B', '#636EFA']
    ).update_layout(yaxis_title='Percentage', xaxis_title='State')

    # Insight 4: Income distribution by loan performance
    insights['Income Distribution'] = px.box(
        df,
        x='Good/Bad Loan',
        y='TOTAL_INCOME',
        title='Income Distribution by Loan Performance',
        color='Good/Bad Loan',
        labels={'TOTAL_INCOME': 'Total Income', 'Good/Bad Loan': 'Loan Performance'}
    )

    # Insight 5: Default rates during COVID vs non-COVID
    covid_default = df.groupby('Covid_Period(Default)')['Good/Bad Loan'].value_counts(normalize=True).unstack() * 100
    insights['COVID Default Rates'] = px.bar(
        covid_default,
        title='Default Rates During COVID vs Non-COVID Periods',
        barmode='group',
        labels={'value': 'Percentage', 'Covid_Period(Default)': 'COVID Period'},
        color_discrete_sequence=['#EF553B', '#636EFA']
    ).update_layout(yaxis_title='Percentage', xaxis_title='COVID Period')

    return insights
