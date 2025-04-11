import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

# Load the pre-trained DBSCAN model
dbscan = joblib.load("dbscan.pkl")

# Streamlit UI
st.title("Customer Segmentation using DBSCAN")

# Selection: Upload File or Enter Data Manually
option = st.radio("Select Input Method:", ["Upload Excel File", "Enter Customer Data Manually"])

# Initialize session state for manually entered data
if "manual_data" not in st.session_state:
    st.session_state.manual_data = []

# Required columns
required_columns = ['Age', 'Education', 'Marital_Status', 'Income', 'Recency', 'Total_Spent', 'Total_Purchases']

# Option 1: Upload Excel File
if option == "Upload Excel File":
    st.subheader("Upload an Excel File")
    uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])

    if uploaded_file:
        df = pd.read_excel(uploaded_file)

        # Check if required columns are present
        if all(col in df.columns for col in required_columns):
            st.success("File uploaded successfully!")
        else:
            st.error(f"Uploaded file must contain columns: {', '.join(required_columns)}")
            df = None
    else:
        df = None

# Option 2: Enter Data Manually
elif option == "Enter Customer Data Manually":
    st.subheader("Enter Customer Data")

    with st.form("manual_entry_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", min_value=1, step=1)
            education = st.selectbox("Education Level", [0, 1], format_func=lambda x: "Lower Education-0" if x == 0 else "Higher Education-1")
            marital_status = st.selectbox("Marital Status", [0, 1], format_func=lambda x: "Single-0" if x == 0 else "Married-1")

        with col2:
            income = st.number_input("Income", step=1000)
            recency = st.number_input("Recency (days)", step=1)

        with col3:
            total_spent = st.number_input("Total Spent", step=100)
            total_purchases = st.number_input("Total Purchases", step=1)

        submit_button = st.form_submit_button("Add Customer")

    # Store manually entered data
    if submit_button:
        new_entry = [age, education, marital_status, income, recency, total_spent, total_purchases]
        st.session_state.manual_data.append(new_entry)

    # Display manually entered data
    if st.session_state.manual_data:
        st.subheader("Manually Entered Customers")
        manual_df = pd.DataFrame(st.session_state.manual_data, columns=required_columns)
        st.dataframe(manual_df)

        df = manual_df
    else:
        df = None

# Process Data (either from Excel or Manual Entry)
if df is not None:
    st.subheader("Segmentation Results")

    # Apply Min-Max Scaling before clustering
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[required_columns])
    scaled_df = pd.DataFrame(scaled_data, columns=required_columns)

    # Apply DBSCAN clustering
    scaled_df['Cluster'] = dbscan.fit_predict(scaled_df)

    # Handle noise points (-1)
    if -1 in scaled_df['Cluster'].values:
        valid_clusters = scaled_df[scaled_df['Cluster'] != -1]['Cluster'].unique()

        if len(valid_clusters) > 0:
            # Assign noise points to the nearest valid cluster
            nbrs = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(scaled_df[scaled_df['Cluster'] != -1].drop(columns=['Cluster']))

            for idx in scaled_df[scaled_df['Cluster'] == -1].index:
                point = scaled_df.loc[idx, :].drop(labels=['Cluster']).values.reshape(1, -1)
                nearest_cluster_idx = nbrs.kneighbors(point, return_distance=False)[0][0]
                nearest_cluster = scaled_df.iloc[nearest_cluster_idx]['Cluster']
                scaled_df.at[idx, 'Cluster'] = nearest_cluster
        else:
            scaled_df['Cluster'] = 0

    # Convert clusters to sequential numbering (1, 2, 3, ...)
    unique_clusters = sorted(scaled_df['Cluster'].unique())
    cluster_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_clusters, start=1)}
    scaled_df['Cluster'] = scaled_df['Cluster'].map(cluster_mapping)

    # Attach clusters back to original (unscaled) data
    df['Cluster'] = scaled_df['Cluster']

    # Display results
    st.write("Segmented Customer Data:")
    st.dataframe(df)

    # Save clustered data to an Excel file
    excel_filename = "customer_segmented.xlsx"
    df.to_excel(excel_filename, index=False)

    # Provide download link
    with open(excel_filename, "rb") as file:
        st.download_button(label="Download Clustered Data (Excel)", data=file, file_name=excel_filename, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
