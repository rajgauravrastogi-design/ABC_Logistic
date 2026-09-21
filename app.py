import streamlit as st
import joblib
import pandas as pd

# Load the trained Logistic Regression model
logi_model = joblib.load('logi.sav')

st.set_page_config(page_title="Delivery Delay Prediction App", layout="centered")

st.title('Delivery Delay Prediction App')
st.write('Enter the features below to predict if there will be a delivery delay.')

# Input fields for features (based on X.columns from the notebook)
# 'Delivery_Distance', 'Traffic_Congestion', 'Weather_Condition',
# 'Delivery_Slot', 'Driver_Experience', 'Num_Stops', 'Vehicle_Age',
# 'Road_Condition_Score', 'Package_Weight', 'Fuel_Efficiency',
# 'Warehouse_Processing_Time'

with st.form("prediction_form"):
    st.header("Delivery Parameters")
    delivery_distance = st.slider('Delivery Distance (km)', min_value=0.0, max_value=50.0, value=25.0, step=0.1)
    traffic_congestion = st.slider('Traffic Congestion (1-5, 5 being highest)', min_value=1, max_value=5, value=3, step=1)
    weather_condition = st.slider('Weather Condition (1-3, 3 being worst)', min_value=1, max_value=3, value=2, step=1)
    delivery_slot = st.selectbox('Delivery Slot (1=Morning, 2=Afternoon, 3=Evening)', [1, 2, 3])
    driver_experience = st.slider('Driver Experience (years)', min_value=0, max_value=20, value=5, step=1)
    num_stops = st.slider('Number of Stops', min_value=1, max_value=10, value=3, step=1)
    vehicle_age = st.slider('Vehicle Age (years)', min_value=0, max_value=15, value=3, step=1)
    road_condition_score = st.slider('Road Condition Score (1-5, 5 being best)', min_value=1, max_value=5, value=3, step=1)
    package_weight = st.slider('Package Weight (kg)', min_value=0.1, max_value=50.0, value=10.0, step=0.1)
    fuel_efficiency = st.slider('Fuel Efficiency (km/L)', min_value=5.0, max_value=20.0, value=10.0, step=0.1)
    warehouse_processing_time = st.slider('Warehouse Processing Time (minutes)', min_value=10, max_value=300, value=120, step=5)

    submitted = st.form_submit_button("Predict Delivery Delay")

    if submitted:
        # Create a DataFrame from the input values
        features = pd.DataFrame([[delivery_distance,
                                  traffic_congestion,
                                  weather_condition,
                                  delivery_slot,
                                  driver_experience,
                                  num_stops,
                                  vehicle_age,
                                  road_condition_score,
                                  package_weight,
                                  fuel_efficiency,
                                  warehouse_processing_time]],
                                columns=['Delivery_Distance', 'Traffic_Congestion', 'Weather_Condition',
                                         'Delivery_Slot', 'Driver_Experience', 'Num_Stops', 'Vehicle_Age',
                                         'Road_Condition_Score', 'Package_Weight', 'Fuel_Efficiency',
                                         'Warehouse_Processing_Time'])

        # Make prediction
        prediction = logi_model.predict(features)
        prediction_proba = logi_model.predict_proba(features)

        st.subheader("Prediction Result:")
        if prediction[0] == 1:
            st.error("**Delay Predicted!**")
            st.write(f"Probability of Delay: {prediction_proba[0][1]*100:.2f}%")
            st.write(f"Probability of No Delay: {prediction_proba[0][0]*100:.2f}%")
        else:
            st.success("**No Delay Predicted!**")
            st.write(f"Probability of No Delay: {prediction_proba[0][0]*100:.2f}%")
            st.write(f"Probability of Delay: {prediction_proba[0][1]*100:.2f}%")
