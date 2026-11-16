import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load('predictor.pkl')

def predict_quality(data_array):
    prediction = model.predict(data_array.reshape(1, -1))
    return prediction[0]

# Page configuration
st.set_page_config(page_title="Wine Quality Predictor", page_icon="🍷", layout="centered")

# Header
st.title("🍷 Wine Quality Predictor")
st.markdown("""
Use this tool to predict the quality of your wine based on its chemical properties. 
Enter the data below and click **Predict** to get started.
""")

# Input Form
with st.form("wine_form"):
    st.subheader("🧪 Chemical Properties")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fixed_acidity = st.number_input('Fixed Acidity', min_value=0.0, value=7.0, step=0.1)
        volatile_acidity = st.number_input('Volatile Acidity', min_value=0.0, value=0.5, step=0.01)
        citric_acid = st.number_input('Citric Acid', min_value=0.0, value=0.2, step=0.01)
        residual_sugar = st.number_input('Residual Sugar', min_value=0.0, value=2.0, step=0.1)
        chlorides = st.number_input('Chlorides', min_value=0.0, value=0.08, step=0.001)
        
    with col2:
        free_sulfur_dioxide = st.number_input('Free Sulfur Dioxide', min_value=0.0, value=15.0, step=1.0)
        total_sulfur_dioxide = st.number_input('Total Sulfur Dioxide', min_value=0.0, value=40.0, step=1.0)
        density = st.number_input('Density', min_value=0.0, value=0.99, step=0.0001)
        pH = st.number_input('pH', min_value=0.0, value=3.3, step=0.01)
        sulphates = st.number_input('Sulphates', min_value=0.0, value=0.6, step=0.01)
        alcohol = st.number_input('Alcohol', min_value=0.0, value=10.0, step=0.1)

    submitted = st.form_submit_button("Predict Quality", type="primary")

# Result Display
if submitted:
    input_data = np.array([
        fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, 
        free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol
    ])
    
    quality = predict_quality(input_data)
    
    st.markdown("---")
    if quality >= 6:
        st.success(f"### ✨ High Quality Wine! \nPredicted Quality Score: {quality}")
    else:
        st.warning(f"### 🍷 Average/Low Quality Wine \nPredicted Quality Score: {quality}")

# Footer
st.markdown("---")
st.caption("Developed by bruzz-bruzz")

