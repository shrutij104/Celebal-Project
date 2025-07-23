import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt # Add this line
import seaborn as sns # Add this line
# import lifetimes
# # print(lifetimes.__version__)
# from lifetimes import BetaGeoFitter, GammaGammaFitter
# import pickle
# import os
# Assuming necessary libraries (streamlit, pandas, lifetimes, pickle, os) are imported

from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data # Need this for refitting
import pickle
import os


# --- Load Data for Refitting (within the model loading cache) ---
# Use a separate cached function to load the full dataset or summary data needed for refitting
@st.cache_data
def load_data_for_refitting():
    """Loads the full dataset or summary data needed to refit lifetimes models."""
    try:
        # Assuming you need the original transactional df or the full summary_df
        # Let's reload the full summary_df as it contains freq=0 customers for BGF
        # and filtered for GGF
        full_summary_df = pd.read_csv('clv_summary_data.csv', index_col='Customer ID') # Assuming this file contains the full summary data
        summary_df_filtered = full_summary_df[full_summary_df['frequency'] > 0].copy()
        return full_summary_df, summary_df_filtered
    except FileNotFoundError:
        st.error("Error: 'clv_summary_data.csv' not found. Cannot refit models.")
        return pd.DataFrame(), pd.DataFrame()
    except Exception as e:
        st.error(f"An error occurred while loading data for refitting: {e}")
        return pd.DataFrame(), pd.DataFrame()


# --- Load and Refit Models ---
@st.cache_resource # Use cache_resource for models
def load_and_refit_models():
    """Loads data and refits lifetimes models for prediction."""

    full_summary_df, summary_df_filtered = load_data_for_refitting()

    if full_summary_df.empty or summary_df_filtered.empty:
        return None, None # Cannot proceed if data loading failed

    try:
        # Initialize models
        bgf = BetaGeoFitter(penalizer_coef=0.0) # Use the same penalizer as training
        ggf = GammaGammaFitter(penalizer_coef=0.0) # Use the same penalizer as training

        # Refit models with the loaded data
        bgf.fit(full_summary_df['frequency'], full_summary_df['recency'], full_summary_df['T'])
        ggf.fit(summary_df_filtered['frequency'], summary_df_filtered['monetary_value'])

        st.sidebar.success("Models refitted successfully!")
        return bgf, ggf

    except Exception as e:
        st.error(f"An error occurred while refitting models: {e}")
        return None, None

# Call the new function to get the refitted models
bgf_model, ggf_model = load_and_refit_models()


# --- Create a Prediction Function (remains mostly the same, uses bgf_model, ggf_model) ---
# ... (the predict_clv function code you added previously goes here) ...
# Ensure predict_clv uses bgf_model and ggf_model which are now refitted instances

# ... rest of your Streamlit app code ...
# Import other libraries you might need, e.g., lifetimes

# ... rest of your code
# Import other libraries you might need, e.g., matplotlib, seaborn, lifetimes

# Set the page configuration (optional)
st.set_page_config(layout="wide", page_title="Customer Lifetime Value Dashboard")

# --- Add a title and introductory text ---
st.title("Customer Lifetime Value (CLV) Analysis Dashboard")
st.write("This dashboard presents the results of the CLV prediction and customer segmentation analysis.")

# --- Data Loading (will add code here in the next step) ---
# @st.cache_data # Use caching for efficient data loading
# --- Data Loading ---
@st.cache_data
def load_data():
    """Loads the prepared CLV summary data."""
    try:
        # Ensure 'clv_summary_data.csv' is in the same directory as your app.py
        # Or provide the correct relative or absolute path
        data = pd.read_csv('clv_summary_data.csv', index_col='Customer ID')
        return data
    except FileNotFoundError:
        st.error("Error: 'clv_summary_data.csv' not found. Make sure the file is in the same directory as app.py")
        return pd.DataFrame() # Return empty DataFrame on error

summary_df_filtered = load_data()

# Check if data loaded successfully and is not empty
if not summary_df_filtered.empty:
    st.sidebar.success("Data loaded successfully!")
    # Display a snippet of the loaded data in the sidebar for verification (optional)
    # st.sidebar.subheader("Data Preview:")
    # st.sidebar.dataframe(summary_df_filtered.head())
else:
    st.sidebar.error("Failed to load data. Please check the file location and name.")

# Assuming summary_df_filtered is loaded and available

# --- Sidebar for Filters ---
st.sidebar.header("Filter Data")

# Get unique segments and countries (add an 'All' option)
all_segments = ['All'] + list(summary_df_filtered['CLV_Segment'].unique())
all_countries = ['All'] + list(summary_df_filtered['Country'].unique())

# Create selectboxes
selected_segment = st.sidebar.selectbox(
    "Select CLV Segment:",
    all_segments
)

selected_country = st.sidebar.selectbox(
    "Select Country:",
    all_countries
)

# --- Apply Filters ---
filtered_df = summary_df_filtered.copy()

if selected_segment != 'All':
    filtered_df = filtered_df[filtered_df['CLV_Segment'] == selected_segment]

if selected_country != 'All':
    filtered_df = filtered_df[filtered_df['Country'] == selected_country]

# Display the number of customers after filtering (optional)
st.sidebar.info(f"Showing data for {len(filtered_df)} customers")

# Now filtered_df contains the data based on user selections.
# You will use this filtered_df for displaying metrics, tables, and visualizations in the main area.


# Assuming filtered_df is available after applying filters

# --- Display Metrics and Tables ---
st.header("CLV and Segment Insights")

if not filtered_df.empty:
    # Display overall metrics for the filtered data
    st.subheader("Overall Metrics (Filtered Data)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Average Predicted CLV", f"€{filtered_df['predicted_clv'].mean():,.2f}")
    col2.metric("Average Frequency", f"{filtered_df['frequency'].mean():.2f}")
    col3.metric("Average Recency", f"{filtered_df['recency'].mean():.2f} days")
    col4.metric("Average Monetary Value", f"€{filtered_df['monetary_value'].mean():,.2f}")

    # Display segment characteristics table (if not filtering by a single segment)
    if selected_segment == 'All':
        st.subheader("Characteristics by CLV Segment (Filtered Data)")
        # Calculate and display segment means for the filtered data
        segment_means_filtered = filtered_df.groupby('CLV_Segment')[['frequency', 'recency', 'monetary_value', 'predicted_clv']].mean().reset_index()
        # Ensure correct order if needed, similar to notebook
        segment_order = ['Low-Value', 'Medium-Value', 'High-Value']
        segment_means_filtered['CLV_Segment'] = pd.Categorical(segment_means_filtered['CLV_Segment'], categories=segment_order, ordered=True)
        segment_means_filtered = segment_means_filtered.sort_values('CLV_Segment')

        st.dataframe(segment_means_filtered.set_index('CLV_Segment'))

    # Display a table of the top customers by predicted CLV (optional)
    st.subheader("Top Customers by Predicted CLV (Filtered Data)")
    st.dataframe(filtered_df.sort_values(by='predicted_clv', ascending=False).head(10)[['predicted_clv', 'frequency', 'recency', 'monetary_value', 'CLV_Segment', 'Country']])

else:
    st.warning("No data matches the selected filters.")


# Assuming filtered_df is available after applying filters
# Assuming segment_order and segment_labels are defined (from the segmentation step)
# Assuming top_countries is defined (from the visualization step in notebook)

# --- Create Visualizations ---
st.header("Visual Analysis")

if not filtered_df.empty:
    # --- Average RFM and Predicted CLV by Segment (Bar Plots) ---
    st.subheader("Average Characteristics by CLV Segment")

    # Ensure segment_means_filtered is calculated based on filtered_df
    segment_means_filtered = filtered_df.groupby('CLV_Segment')[['frequency', 'recency', 'monetary_value', 'predicted_clv']].mean().reset_index()
    # Ensure correct order if needed
    segment_order = ['Low-Value', 'Medium-Value', 'High-Value'] # Define or ensure this is available
    segment_means_filtered['CLV_Segment'] = pd.Categorical(segment_means_filtered['CLV_Segment'], categories=segment_order, ordered=True)
    segment_means_filtered = segment_means_filtered.sort_values('CLV_Segment')


    fig1, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig1.suptitle('Average Characteristics by CLV Segment (Filtered Data)', y=1.02, fontsize=16)

    sns.barplot(x='CLV_Segment', y='frequency', data=segment_means_filtered, ax=axes[0, 0], palette='viridis')
    axes[0, 0].set_title('Average Frequency by Segment')
    axes[0, 0].set_xlabel('')
    axes[0, 0].set_ylabel('Average Frequency')

    sns.barplot(x='CLV_Segment', y='recency', data=segment_means_filtered, ax=axes[0, 1], palette='viridis')
    axes[0, 1].set_title('Average Recency by Segment')
    axes[0, 1].set_xlabel('')
    axes[0, 1].set_ylabel('Average Recency (Days)')

    sns.barplot(x='CLV_Segment', y='monetary_value', data=segment_means_filtered, ax=axes[1, 0], palette='viridis')
    axes[1, 0].set_title('Average Monetary Value by Segment')
    axes[1, 0].set_xlabel('')
    axes[1, 0].set_ylabel('Average Monetary Value')

    sns.barplot(x='CLV_Segment', y='predicted_clv', data=segment_means_filtered, ax=axes[1, 1], palette='viridis')
    axes[1, 1].set_title('Average Predicted CLV by Segment')
    axes[1, 1].set_xlabel('')
    axes[1, 1].set_ylabel('Average Predicted CLV')

    plt.tight_layout()
    st.pyplot(fig1) # Display the figure in Streamlit
    plt.close(fig1) # Close the figure to free memory

    # --- Distribution of Country by Segment (Stacked Bar Chart) ---
    st.subheader("Distribution of Top Countries within CLV Segments")

    if selected_country == 'All': # Only show this plot if not filtering by a single country
        # Calculate the proportion of each country within each segment for filtered data
        country_segment_proportion_filtered = pd.crosstab(filtered_df['CLV_Segment'], filtered_df['Country'], normalize='index')

        # Select top N countries based on overall representation in the filtered data
        # Adjust N as needed, or base on total data if preferred
        top_countries_filtered = filtered_df['Country'].value_counts().nlargest(10).index.tolist()
        country_segment_proportion_top_filtered = country_segment_proportion_filtered[top_countries_filtered]

        fig2 = country_segment_proportion_top_filtered.plot(kind='bar', stacked=True, figsize=(12, 7), colormap='viridis')
        plt.title('Proportion of Top Countries within CLV Segments (Filtered Data)')
        plt.xlabel('CLV Segment')
        plt.ylabel('Proportion within Segment')
        plt.xticks(rotation=0)
        plt.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        st.pyplot(fig2.figure) # Display the figure in Streamlit
        plt.close(fig2.figure) # Close the figure

    else:
        st.info("Country distribution plot is shown when 'All' countries are selected.")


else:
    st.warning("No data available to generate visualizations for the selected filters.")

# --- Add sections for different insights ---
# st.header("Overall CLV Metrics")
# st.header("Segment Characteristics")
# st.header("Geographic Distribution")

# --- Placeholder for visualizations and tables (will add code here later) ---


# --- Running the App ---
# To run this Streamlit app, save the code as a Python file (e.g., app.py)
# Open your terminal or command prompt, navigate to the directory where you saved the file,
# and run the command: streamlit run app.py
# Assuming necessary libraries (streamlit, pandas, lifetimes) are imported at the top


# --- Load Saved Model Parameters ---
@st.cache_resource # Use cache_resource for models/parameters that don't change
def load_models():
    """Loads the saved lifetimes model parameters."""
    models_dir = 'models'
    try:
        with open(os.path.join(models_dir, 'bgf_params.pkl'), 'rb') as f:
            bgf_params = pickle.load(f)

        with open(os.path.join(models_dir, 'ggf_params.pkl'), 'rb') as f:
            ggf_params = pickle.load(f)

        # Create new model instances and set parameters
        bgf = BetaGeoFitter()
        bgf.params_ = bgf_params

        ggf = GammaGammaFitter()
        ggf.params_ = ggf_params

        st.sidebar.success("Models loaded successfully!")
        return bgf, ggf

    except FileNotFoundError:
        st.error("Error: Model parameter files not found. Make sure 'models' directory and .pkl files are in the same directory as app.py")
        return None, None
    except Exception as e:
        st.error(f"An error occurred while loading models: {e}")
        return None, None

bgf_model, ggf_model = load_models()

# --- Create a Prediction Function ---
# This function will take customer data and return predicted CLV
if bgf_model and ggf_model:
    def predict_clv(customer_data):
        """
        Predicts CLV for new customer data using loaded models.
        customer_data should be a DataFrame with columns:
        'frequency', 'recency', 'T', 'monetary_value'
        """
        # Ensure required columns are present
        required_cols = ['frequency', 'recency', 'T', 'monetary_value']
        if not all(col in customer_data.columns for col in required_cols):
            st.error(f"Input data must contain the following columns: {required_cols}")
            return None

        # Filter data for Gamma-Gamma prediction (frequency > 0)
        customer_data_filtered = customer_data[customer_data['frequency'] > 0].copy()

        if not customer_data_filtered.empty:
             # Predict future purchases (BG/NBD)
            customer_data['predicted_purchases'] = bgf_model.predict(
                12, # Assuming 12 periods prediction, match your training
                customer_data['frequency'],
                customer_data['recency'],
                customer_data['T']
            )

            # Predict average monetary value (Gamma-Gamma) for freq > 0
            customer_data_filtered['predicted_monetary_value'] = ggf_model.conditional_expected_average_profit(
                customer_data_filtered['frequency'],
                customer_data_filtered['monetary_value']
            )

            # Merge predicted monetary value back (keep all customers)
            predicted_data = customer_data[['predicted_purchases']].merge(
                customer_data_filtered['predicted_monetary_value'],
                left_index=True,
                right_index=True,
                how='left'
            )

            # For customers with frequency=0, monetary prediction is NaN, set to 0 for CLV
            predicted_data['predicted_monetary_value'].fillna(0, inplace=True)


            # Calculate predicted CLV
            predicted_data['predicted_clv'] = predicted_data['predicted_purchases'] * predicted_data['predicted_monetary_value']

            return predicted_data[['predicted_purchases', 'predicted_monetary_value', 'predicted_clv']]

        else:
            # Handle case where all input customers have frequency = 0
            st.info("All input customers have frequency 0. Gamma-Gamma model not applicable.")
            customer_data['predicted_purchases'] = bgf_model.predict(
                12, # Assuming 12 periods prediction
                customer_data['frequency'],
                customer_data['recency'],
                customer_data['T']
            )
            customer_data['predicted_monetary_value'] = 0 # Monetary value is 0 for freq=0
            customer_data['predicted_clv'] = 0 # CLV is 0 for freq=0
            return customer_data[['predicted_purchases', 'predicted_monetary_value', 'predicted_clv']]


else:
    st.error("Models not loaded. Cannot make predictions.")
    predict_clv = None # Define predict_clv as None if models failed to load


# Now you can call predict_clv(some_dataframe) later in your app to make predictions.
# Assuming filtered_df is available
# Assuming predict_clv function is defined and models loaded

if not filtered_df.empty and predict_clv:
    st.subheader("Predicted CLV for Filtered Customers")

    # Make predictions for the filtered data using the predict_clv function
    # Need to pass only the relevant RFM-T columns
    prediction_input_data = filtered_df[['frequency', 'recency', 'T', 'monetary_value']]
    predicted_results = predict_clv(prediction_input_data)

    if predicted_results is not None:
        # Merge predictions back with filtered_df for display
        filtered_df_with_predictions = filtered_df.merge(
            predicted_results,
            left_index=True,
            right_index=True,
            how='left'
        )

        # Display a table with key info and predictions
        st.dataframe(filtered_df_with_predictions[['predicted_clv', 'predicted_purchases', 'predicted_monetary_value', 'frequency', 'recency', 'T', 'monetary_value', 'CLV_Segment', 'Country']].head()) # Display head or full table

        # You could also visualize distribution of predicted CLV for filtered data
        # st.subheader("Distribution of Predicted CLV (Filtered Data)")
        # fig_pred_clv, ax_pred_clv = plt.subplots()
        # sns.histplot(filtered_df_with_predictions['predicted_clv'], kde=True, ax=ax_pred_clv)
        # ax_pred_clv.set_title("Distribution of Predicted CLV")
        # st.pyplot(fig_pred_clv)
        # plt.close(fig_pred_clv)


    # --- Optional: Manual Input for Prediction ---
    st.subheader("Get CLV Prediction for a Hypothetical Customer")
    with st.form("predict_form"):
        st.write("Enter customer RFM-T data:")
        freq_input = st.number_input("Frequency:", min_value=0.0, value=1.0)
        rec_input = st.number_input("Recency (days):", min_value=0.0, value=30.0)
        t_input = st.number_input("Customer Age (T in days):", min_value=0.0, value=180.0)
        monetary_input = st.number_input("Monetary Value:", min_value=0.0, value=50.0)
        # Note: Country is not used in the lifetimes prediction itself, only RFM-T

        predict_button = st.form_submit_button("Predict CLV")

        if predict_button:
            # Create a DataFrame for the input data
            input_data_df = pd.DataFrame({
                'frequency': [freq_input],
                'recency': [rec_input],
                'T': [t_input],
                'monetary_value': [monetary_input]
            })

            # Get prediction using the function
            manual_prediction = predict_clv(input_data_df)

            if manual_prediction is not None:
                st.subheader("Prediction Results:")
                st.write(manual_prediction)


else:
    if not predict_clv:
         st.error("Prediction function not available because models failed to load.")
    # Message for empty filtered_df is already handled in the metrics section
