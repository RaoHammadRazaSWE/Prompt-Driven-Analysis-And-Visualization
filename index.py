# install and import libraries.
import pandas as pd
import streamlit as st
import os 
from dotenv import load_dotenv
from IPython.display import Markdown , display
import matplotlib.pyplot as plt
import seaborn as sns 
import plotly.graph_objects as go
import os
from pandasai import Agent


os.environ["PANDASAI_API_KEY"] = "Enter API KEY"


# Function to save sidebar outputs to session state
def save_sidebar_outputs():
    if 'df_shape' not in st.session_state:
        st.session_state.df_shape = None
    if 'df_dtypes' not in st.session_state:
        st.session_state.df_dtypes = None
    if 'df_columns' not in st.session_state:
        st.session_state.df_columns = None
    if 'df_describe' not in st.session_state:
        st.session_state.df_describe = None
    if 'cols_to_drop' not in st.session_state:
        st.session_state.cols_to_drop = None
    if 'original_df' not in st.session_state:
        st.session_state.original_df = None

st.title("Prompt-driven analysis with pandas")

uploaded_file = st.file_uploader("Upload your CSV file for analysis", type=["csv", "xlsx"])

if uploaded_file is not None:
    if 'original_df' not in st.session_state or st.session_state.original_df is None:
        df = pd.read_csv(uploaded_file)
        st.session_state.original_df = df.copy()
    else:
        df = st.session_state.original_df.copy()
    st.write(df.head(3))


    save_sidebar_outputs()

    about_data_button = st.sidebar.button("About the data")
    if about_data_button:
        st.session_state.df_shape = df.shape
        st.session_state.df_dtypes = df.dtypes
        st.session_state.df_columns = df.columns
        st.session_state.df_isnull = df.isnull().sum()
        st.session_state.df_unique = df.nunique()

    if 'df_shape' in st.session_state and st.session_state.df_shape is not None:
        st.sidebar.subheader("Shape of the Data:")
        st.sidebar.text(st.session_state.df_shape)
    if 'df_dtypes' in st.session_state and st.session_state.df_dtypes is not None:
        st.sidebar.subheader("Data Types:")
        st.sidebar.write(st.session_state.df_dtypes)
    if 'df_columns' in st.session_state and st.session_state.df_columns is not None:
        st.sidebar.subheader("Columns in the Dataset:")
        st.sidebar.write(st.session_state.df_columns)
    if 'df_isnull' in st.session_state and st.session_state.df_isnull is not None:
        st.sidebar.subheader("Null values in Data:")
        st.sidebar.write(st.session_state.df_isnull)
    if 'df_unique' in st.session_state and st.session_state.df_unique is not None:
        st.sidebar.subheader("Unique values in Data :")
        st.sidebar.write(st.session_state.df_unique)
    
    # Step 1: Remove selected columns
    if st.session_state.cols_to_drop is None:
        st.session_state.cols_to_drop = []
    cols_to_drop = st.multiselect("Step 1: Select column to drop", options=df.columns, default=[col for col in st.session_state.cols_to_drop if col in df.columns])
    if cols_to_drop:
        if st.button("Remove columns"):
            df = df.drop(columns=cols_to_drop)
            st.session_state.cols_to_drop = cols_to_drop
            st.session_state.original_df = df.copy()  # Update original DataFrame
            st.success(f"Columns {', '.join(cols_to_drop)} dropped successfully!")
        else:
            st.warning("Please select columns to drop")
    if st.button("see result"):
        st.write(df.columns)
    # Step 2: Remove null values from selected columns
    col_to_drop_null = st.multiselect("Step 2: Select column to drop null values", options=df.columns)
    if col_to_drop_null:
        if st.button("Remove null values"):
            for col in col_to_drop_null:
                if col in df.columns:
                    df = df.dropna(subset=[col])
                    st.success(f"Null values in '{col}' dropped successfully!")
            st.session_state.original_df = df.copy()  # Update original DataFrame
        if st.button("see_result"):
            st.write(df.isnull().sum())
    # Step 3: Plot bar chart for selected column
    col_to_plot = st.selectbox("Step 3: Select column for histogram plot", options=df.columns)
    if col_to_plot:
        if st.button("Histogram Plot"):
            plt.figure(figsize=(4,2))
            df[col_to_plot].plot(kind='hist')
            plt.xlabel(col_to_plot)
            plt.ylabel('Frequency')
            plt.title(f'Histogram plot of {col_to_plot}')
            st.pyplot(plt)
            st.session_state.original_df = df.copy()  # Update original DataFrame
    # Step 4: Create boxplot for selected column
    col_to_plot_boxplot = st.selectbox("Step 4: Select column for boxplot", options=df.columns)
    if col_to_plot_boxplot:
        if st.button("Boxplot"):
            plt.figure(figsize=(8, 6))
            sns.boxplot(x=df[col_to_plot_boxplot])
            plt.xlabel(col_to_plot_boxplot)
            plt.title(f'Boxplot of {col_to_plot_boxplot}')
            st.pyplot(plt)
            st.session_state.original_df = df.copy()  # Update original DataFrame
        
    # Step 5: Calculate Interquartile Range (IQR) and print it
    if st.button("Calculate and Print IQR"):
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        st.write("Interquartile Range (IQR):")
        st.write(IQR)
    # Step 6: Calculate correlation matrix
    if st.button("Calculate Correlation Matrix"):
        correlation_matrix = df.corr()
        st.write("Correlation Matrix:")
        st.write(correlation_matrix)
    # Step 7: Display bar plot of two variables
    selected_columns = st.multiselect("Step 7: Select two columns for scatter plot", options=df.columns)
    hue_column = st.selectbox("Select a column for hue (optional)", options=df.columns, index=0)
    if len(selected_columns) == 2:
        if st.button("Display scatter Plot"):
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=selected_columns[1], y=selected_columns[0],hue=hue_column, data=df)
            plt.title(f'Line scatter of {selected_columns[0]} vs {selected_columns[1]}')
            plt.legend(title=hue_column)
            st.pyplot(plt)
    # Step 8: Display bar plot of two variables
    selected_column_barplot = st.multiselect("Select two columns for bar plot", options=df.columns)
    st.write("Select Y value First :")
    if len(selected_column_barplot) == 2:
        if st.button("Display Bar Plot"):
            plt.figure(figsize=(10, 5))
            df[selected_column_barplot[0]].value_counts().nlargest(15).plot(kind='bar')
            plt.xlabel(selected_column_barplot[1])
            plt.ylabel(selected_column_barplot[0])
            plt.title(f'Bar Plot of {selected_column_barplot}')
            st.pyplot(plt)
    # Step 9: Display bar plot of two variables
    selected_columns = st.multiselect("Step 7: Select two columns for line plot", options=df.columns)
    hue_options = [None] + list(df.columns)
    hue_column = st.selectbox("Select a column for hue", options=hue_options)
    if len(selected_columns) == 2:
        if st.button("Display lineplot Plot"):
            plt.figure(figsize=(10, 6))
            sns.lineplot(x=selected_columns[1], y=selected_columns[0], hue=hue_column , style=hue_column, data=df)
            plt.title(f'Line scatter of {selected_columns[0]} vs {selected_columns[1]}')
            plt.legend(title=hue_column)
            st.pyplot(plt)

# Step 10: Display pie plot of one variable


# Function to filter columns based on data types
def filter_columns_by_dtype(df, dtype):
    return [col for col, col_dtype in df.dtypes.items() if col_dtype == dtype]

# Step 11: Display Sunburst chart
selected_column_sunburst = st.selectbox("Step 11: Select a hierarchical column for sunburst chart", options=filter_columns_by_dtype(df, 'object'))

if selected_column_sunburst:
    # Get the selected column values
    column_values = df[selected_column_sunburst].dropna().unique().tolist()

    # Ask user to select the column values
    selected_values = st.multiselect(f"Select values for '{selected_column_sunburst}'", options=column_values)

    if selected_values:
        # Filter the DataFrame based on selected values
        filtered_df = df[df[selected_column_sunburst].isin(selected_values)]

        # Prepare data for Sunburst chart
        labels = filtered_df[selected_column_sunburst].tolist()
        parents = [''] * len(labels)  # All nodes are root nodes
        values = [1] * len(labels)  # Assigning a value of 1 to each node

        # Create Sunburst chart
        fig = go.Figure(go.Sunburst(labels=labels, parents=parents, values=values))
        # Update layout for tight margin
        fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))

        # Display the chart
        st.plotly_chart(fig)

            
