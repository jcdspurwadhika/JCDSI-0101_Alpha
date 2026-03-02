import streamlit as st
import joblib
import pandas as pd
import numpy as np

#1. Load the trained model pipeline
pipeline = joblib.load('final_random_forest_pipeline.joblib')

st.title("Bank Marketing Campaign Predictor")
st.write("Enter client details below to predict if they will subscribe to a term deposit.")

#--- INPUT FIELDS ---
#We use inputs that make sense to the user, then convert them later.
age = st.number_input("Age", min_value=18, max_value=100, value=30)
job = st.selectbox("Job", ["management", "technician", "entrepreneur", "blue-collar", "unknown", "retired", "admin.", "services", "self-employed", "unemployed", "housemaid", "student"])
marital = st.selectbox("Marital Status", ["married", "single", "divorced", "unknown"])
education = st.selectbox("Education", ["basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate", "professional.course", "university.degree", "unknown"])
default = st.selectbox("Has Credit in Default?", ["no", "yes", "unknown"])
housing = st.selectbox("Has Housing Loan?", ["no", "yes", "unknown"])
loan = st.selectbox("Has Personal Loan?", ["no", "yes", "unknown"])
contact = st.selectbox("Contact Communication Type", ["cellular", "telephone"])
month = st.selectbox("Last Contact Month", ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])
day_of_week = st.selectbox("Last Contact Day of Week", ["mon", "tue", "wed", "thu", "fri"])
campaign = st.number_input("Number of Contacts during this Campaign", min_value=1, value=1)
pdays = st.number_input("Days Since Last Contact (999 means never)", min_value=0, max_value=999, value=999)
previous = st.number_input("Number of Contacts Before this Campaign", min_value=0, value=0)
poutcome = st.selectbox("Outcome of Previous Marketing Campaign", ["nonexistent", "failure", "success"])
emp_var_rate = st.number_input("Employment Variation Rate", value=1.1)
cons_price_idx = st.number_input("Consumer Price Index", value=93.994)
cons_conf_idx = st.number_input("Consumer Confidence Index", value=-36.4)
euribor3m = st.number_input("Euribor 3 Month Rate", value=4.856)
nr_employed = st.number_input("Number of Employees", value=5191.0)

#--- PREDICTION BUTTON ---
if st.button("Predict Subscription"):
    #1. Create a DataFrame for raw input
    input_data = pd.DataFrame({
        'age': [age],
        'job': [job],
        'marital': [marital],
        'education': [education],
        'default': [default],
        'housing': [housing],
        'loan': [loan],
        'contact': [contact],
        'month': [month],
        'day_of_week': [day_of_week],
        'campaign': [campaign],
        'pdays': [pdays],
        'previous': [previous],
        'poutcome': [poutcome],
        'emp_var_rate': [emp_var_rate],
        'cons_price_idx': [cons_price_idx],
        'cons_conf_idx': [cons_conf_idx],
        'euribor3m': [euribor3m],
        'nr_employed': [nr_employed]
    })
    
    #FEATURE ENGINEERING (Re-creating training features)
    
    #Engineering 'was_contacted_before'
    input_data['was_contacted_before'] = input_data['pdays'].apply(lambda x: 1 if x != 999 else 0)
    
    #Engineering 'age_group'
    #NOTE: You must use the EXACT bins and labels from your notebook here.
    #Below is a common example, please check your notebook for exact values.
    bins = [0, 20, 30, 40, 50, 60, 100]
    labels = ['<20', '20s', '30s', '40s', '50s', '60+']
    input_data['age_group'] = pd.cut(input_data['age'], bins=bins, labels=labels, include_lowest=True)
    
    #Ensure age_group is treated as a categorical variable
    input_data['age_group'] = input_data['age_group'].astype('object') # Often necessary before pipeline
    
    #3. Make Prediction
    prediction = pipeline.predict(input_data)
    probability = pipeline.predict_proba(input_data)[0][1]
    
    #4. Display Result
    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.success(f"Likely to subscribe! (Confidence: {probability:.2%})")
    else:
        st.error(f"Unlikely to subscribe. (Confidence: {1-probability:.2%})")

    
#To run put "streamlit run app.py" on terminal   