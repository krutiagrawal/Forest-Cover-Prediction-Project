import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# Loading the trained model and scaler
model = load_model('forest_cover_model.h5')
scaler = joblib.load('scaler.pkl')

# Streamlit App Layout
st.title("Forest Cover Type Prediction")
st.write("Predict the type of forest cover based on environmental data.")

# Input fields
features = {}
features['Elevation'] = st.number_input('Elevation in meters', value=2000)
features['Aspect'] = st.number_input('Aspect in degrees azimuth', value=50)
features['Slope'] = st.number_input('Slope in degrees', value=10)
features['Horizontal_Distance_To_Hydrology'] = st.number_input('Horizontal Distance to Hydrology', value=100)
features['Vertical_Distance_To_Hydrology'] = st.number_input('Vertical Distance to Hydrology', value=50)
features['Horizontal_Distance_To_Roadways'] = st.number_input('Horizontal Distance to Roadways', value=500)
features['Hillshade_9am'] = st.number_input('Hillshade at 9am', value=200)
features['Hillshade_Noon'] = st.number_input('Hillshade at Noon', value=220)
features['Hillshade_3pm'] = st.number_input('Hillshade at 3pm', value=180)
features['Horizontal_Distance_To_Fire_Points'] = st.number_input('Horizontal Distance to Fire Points', value=300)

# Converting input to DataFrame and scale
input_data = np.array([[features[col] for col in features]])
input_data_scaled = scaler.transform(input_data)

# Prediction button
if st.button('Predict Cover Type'):
    prediction = model.predict(input_data_scaled)
    cover_type = np.argmax(prediction) + 1
    cover_types = {
        1: "Spruce/Fir",
        2: "Lodgepole Pine",
        3: "Ponderosa Pine",
        4: "Cottonwood/Willow",
        5: "Aspen",
        6: "Douglas-fir",
        7: "Krummholz"
    }
    st.write(f"Predicted Forest Cover Type: {cover_types[cover_type]}")
