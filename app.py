import streamlit as st
import pandas as pd
import joblib

from os.path import exists
MODEL_PATH = 'model/'
if not exists(f'{MODEL_PATH}/tf_model.pkl'):
    st.write('Model not found')
else:
    model = joblib.load(f'{MODEL_PATH}/tf_model.pkl')
    st.write('Loading...')

st.subheader("Keyin Engine Parameters:")
Engine_rpm  = st.number_input("Engine rpm", value=791)
Lub_oil_pressure = st.number_input("Lub oil pressure", value=3)
Fuel_pressure = st.number_input("Fuel pressure", value=6)
Coolant_pressure = st.number_input("Coolant pressure", value=2)
Lub_oil_temp = st.number_input("Lub oil temp", value=76)
Coolant_temp = st.number_input("Coolant temp", value=78)

st.divider()

from numpy import load
if st.button("Predict"):
    model = joblib.load(f'{MODEL_PATH}/tf_model.pkl')
    input_data = pd.DataFrame([{
        "Engine rpm": Engine_rpm,
        "Lub oil pressure": Lub_oil_pressure,
        "Fuel pressure": Fuel_pressure,
        "Coolant pressure": Coolant_pressure,
        "lub oil temp": Lub_oil_temp,
        "Coolant temp": Coolant_temp
        }])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    if prediction[0] > 0.5:
        st.info("Engine condition is Good")
    st.write(f"Prediction: {prediction[0]}")
    if prediction[0] < 0.5:
      st.info("Engine condition is normal")
    elif prediction[0] == 1:
      st.info("Engine condition is Faulty")


