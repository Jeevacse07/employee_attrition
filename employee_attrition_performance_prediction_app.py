import streamlit as st
import joblib
import numpy as np

# Load models
attrition_model = joblib.load('attrition_xgboost_model.pkl')
performance_rating_model = joblib.load('performance_rating_xgboost_model.pkl')

# Title
st.title("Employee Performance And Attrition Prediction App")

# Feature definitions
attrition_features = ['overtime', 'maritalstatus_single', 'yearsincurrentrole', 'monthlyincome',
                   'jobrole_sales representative', 'yearswithcurrmanager', 'stockoptionlevel',
                   'jobinvolvement', 'businesstravel_travel_frequently', 'jobsatisfaction',
                   'environmentsatisfaction', 'jobrole_laboratory technician', 
                   'jobrole_research director', 'department_research & development']

performance_rating_features = ['percentsalaryhike','yearsincurrentrole', 'jobrole_manager', 
                  'relationshipsatisfaction', 'environmentsatisfaction','jobinvolvement',
                  'distancefromhome', 'businesstravel_travel_rarely', 'education',
                    'educationfield_technical degree', 'joblevel', 'gender',
                  'overtime','worklifebalance','trainingtimeslastyear']

# Tabs
tabs = st.tabs(["Attrition", "Performance"])

### TAB 1: Attrition
with tabs[0]:
    st.subheader("Employee Attrition Prediction") 

    # Initialize and reset
    for key in attrition_features:
        full_key = f"attrition_{key}"
        st.session_state.setdefault(full_key, 0.0)

    if st.button("Reset Attrition Form"):
        for key in attrition_features:
            st.session_state[f"attrition_{key}"] = 0.0
        st.rerun()

    attrition_values = []
    for key in attrition_features:
        full_key = f"attrition_{key}"
        val = st.number_input(key, key=full_key, value=st.session_state[full_key])
        attrition_values.append(val)

    if st.button("Predict Attrition"):
        input_array = np.array(attrition_values).reshape(1, -1)
        result = attrition_model.predict(input_array)[0]
        st.success("Employee Will leave" if result == 1 else "Employee Will not leave")

### TAB 2: Performance
with tabs[1]:
    st.subheader("Performance Rating Prediction")

    for key in performance_rating_features:
        full_key = f"performance_{key}"
        st.session_state.setdefault(full_key, 0.0)

    if st.button("Reset Performance Rating Form"):
        for key in performance_rating_features:
            st.session_state[f"performance_{key}"] = 0.0
        st.rerun()

    performance_rating_values = []
    for key in performance_rating_features:
        full_key = f"performance_{key}"
        val = st.number_input(key, key=full_key, value=st.session_state[full_key])
        performance_rating_values.append(val)

    if st.button("Predict Employee Performance Rating"):
        input_array = np.array(performance_rating_values).reshape(1, -1)
        result = performance_rating_model.predict(input_array)[0]
        st.success("Employee Good Rating" if result == 1 else "Employee Average Rating")
