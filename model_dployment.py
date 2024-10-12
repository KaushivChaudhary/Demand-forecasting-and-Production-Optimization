# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 16:27:29 2024

@author: Kaushiv
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import pickle
import streamlit as st
import matplotlib.pyplot as plt
from scipy import stats
import joblib

# Load pre-trained models and scalers
modelxgb = pickle.load(open('xgb_model.pkl', 'rb'))
modelse = pickle.load(open('ses_model.pkl', 'rb'))
modellr = pickle.load(open('linear_model.pkl', 'rb'))
model = pickle.load(open('se_model.pkl', 'rb'))

def predict(data):
    prediction = pd.Series(model.predict(data))
    data["Economic Index"] = prediction
    data['Industry Growth Rate (%)'] = prediction  # Fixed incorrect Series creation
    return data     


# Main Streamlit app function
def main():
    st.title("Time Series Forecasting with Confidence Intervals")
    st.sidebar.title("Forecasting")
    
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Forecasting</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.text("")

    uploaded_file = st.sidebar.file_uploader("Choose a file", type=['csv', 'xlsx'], accept_multiple_files=False, key="fileUploader")
    import pandas as pd
    if uploaded_file is not None:
        # Determine the file type
        try:
            # Try reading as CSV
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            # Try reading as Excel
            elif uploaded_file.name.endswith('.xlsx'):
                data = pd.read_excel(uploaded_file)
            else:
                st.error("The file format is not supported. Please upload a CSV or Excel file.")
                return  # Exit if the file type is not supported

            st.success("File uploaded successfully!")
            st.write(data.head())  # Display the first few rows of the dataframe
            
            # Display the first few rows of the dataframe

                # Ensure 'Billing date' column exists before processing
            if 'Billing date' in data.columns:
                data['Billing date'] = pd.to_datetime(data['Billing date'], errors='coerce')
            else:
                st.error("Column 'Billing date' is missing from the uploaded data.")
                return
            
        except Exception as e:
            st.error(f"Error reading the file: {e}")

    # Check if "Predict" button is clicked
    if st.button("Predict"):
        # Preprocessing and Feature Engineering
        # Get dummies for 'Seasonality Factor'   
        # Create a 'Month' column as a Period
        data['Month'] = data['Billing date'].dt.to_period('M')
        
        # Convert 'Month' to Timestamp for further processing if it is Period
        data['Month'] = data['Month'].dt.to_timestamp()
        
        # Check and convert 'Month' to datetime if needed
        #if not pd.api.types.is_datetime64_any_dtype(data['Month']):
            #data['Month'] = pd.to_datetime(data['Month'], errors='coerce')
        
        # Group by 'Month', 'Variant', and 'Seasonality Factor' and aggregate the data
        data_f = data.groupby(['Month', 'Variant', 'Seasonality Factor']).agg({
            'Economic Index': 'sum', # Sum of 'Economic Index' for the month
            'Industry Growth Rate (%)': 'sum'
        }).reset_index()
        
        # If you need to ensure 'Month' is in datetime format after grouping, you can do:
        data_f['Month'] = pd.to_datetime(data_f['Month'])

# Now 'monthly_data' is ready for further analysis or processing

        data_f = pd.get_dummies(data_f, columns=['Seasonality Factor'], drop_first=True)
  # Prediction function


        # Example list of variants you want to filter by
        selected_variants_lr = ['XXX11', 'XXX15', 'XXX18','XXXV5','XXXV9']  # Replace with your desired variants
        selected_variants_xgb=['XXX12','XXX13','XXX17']
        selected_variants_se=['XXXV1','XXXV2','XXXV3','XXXV4']
        var=['XXX11','XXX12','XXX13','XXX17', 'XXX15', 'XXX18','XXXV1','XXXV2','XXXV3','XXXV4','XXXV5','XXXV9']

        # Ensure 'Variant' column exists and handle variant-based prediction
        if 'Industry Growth Rate (%)' in data_f.columns:
            variant = data['Variant'].iloc[0]  # Assuming one variant per uploaded file

            # Predict based on variant
            if variant in selected_variants_lr:
                res = modellr.predict(data)
            elif variant in selected_variants_xgb:
                res = modelxgb.predict(data)
            elif variant in selected_variants_se:
                res = modelse.predict(data)
            else:
                st.error("Variant not found in model lists.")
                return
        else:
            st.error("Variant column is missing from the data.")
            return

        # Use SES model for Economic Growth or other specific columns if applicable
        growth_forecast = None
        if 'Economic Index' in data.columns:
            growth_forecast = model.predict(data)  # Assuming the SES model predicts based on existing data
        else:
            st.warning("Economic Index column is missing, skipping SES model prediction.")

        # Display the predictions
        st.write("Model Prediction Results:")
        st.write(res)

        if growth_forecast is not None:
            st.write("Data with Economic Growth Predictions:")
            st.write(growth_forecast)

        # Compute confidence intervals
        def compute_confidence_intervals(predictions, stdev, confidence_level):
            z = stats.norm.ppf((1 + confidence_level) / 2)
            lower = predictions - z * stdev
            upper = predictions + z * stdev
            return lower, upper

        # Calculate standard deviation for the residuals
        stdev = np.std(data['Industry Growth Rate (%)'] - res)
        if growth_forecast is not None:
            stdev2 = np.std(data['Economic Index'] - growth_forecast)
        
        # Calculate confidence intervals (90% and 95%)
        lower_90, upper_90 = compute_confidence_intervals(res, stdev, 0.90)
        lower_95, upper_95 = compute_confidence_intervals(res, stdev, 0.95)

        if growth_forecast is not None:
            lower2_90, upper2_90 = compute_confidence_intervals(growth_forecast, stdev2, 0.90)
            lower2_95, upper2_95 = compute_confidence_intervals(growth_forecast, stdev2, 0.95)

        # Display predictions and confidence intervals
        results = pd.DataFrame({
            'Actual': data['Industry Growth Rate (%)'],
            'Predicted': res,
            'Lower 90%': lower_90,
            'Upper 90%': upper_90,
            'Lower 95%': lower_95,
            'Upper 95%': upper_95
        })

        st.subheader("Predictions vs Actual with Confidence Intervals")
        st.write(results)

        # Plot predictions and confidence intervals
        fig, ax = plt.subplots()
        ax.plot(data.index, data['Industry Growth Rate (%)'], label='Actual', color='blue')
        ax.plot(data.index, res, label='Predicted', color='red')
        ax.fill_between(data.index, lower_90, upper_90, color='orange', alpha=0.3, label='90% CI')
        ax.fill_between(data.index, lower_95, upper_95, color='green', alpha=0.2, label='95% CI')
        ax.legend()
        st.pyplot(fig)

if __name__ == '__main__':
    main()
