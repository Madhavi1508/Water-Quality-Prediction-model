import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import yaml
from utils import validate_inputs

# Load model, scaler, and config
model = load_model('water_quality_model.h5')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Streamlit configuration
st.set_page_config(page_title="Water Quality Prediction", page_icon="static/images/favicon.ico")

# Apply custom CSS
st.markdown('<style>' + open('static/css/style.css').read() + '</style>', unsafe_allow_html=True)

# App title and description
st.title('Water Quality Prediction')
st.markdown('Enter water quality parameters to check if the water is potable.')

# Input form
with st.form(key='input_form'):
    user_inputs = []
    for feature in config['features']:
        user_inputs.append(st.number_input(
            feature['name'],
            min_value=float(feature['min']),
            max_value=float(feature['max']),
            value=float(feature['default']),
            step=0.1
        ))
    submit_button = st.form_submit_button(label='Predict Potability')

# Prediction
if submit_button:
    if validate_inputs(user_inputs, config['features']):
        input_data = np.array([user_inputs])
        input_scaled = scaler.transform(input_data)
        prediction_prob = model.predict(input_scaled, verbose=0)[0][0]
        result = 'Yes' if prediction_prob > 0.5 else 'No'
        if result == 'Yes':
            st.markdown('<p class="success">Potability: Yes</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="error">Potability: No</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="error">Please enter valid values within the specified ranges.</p>', unsafe_allow_html=True)