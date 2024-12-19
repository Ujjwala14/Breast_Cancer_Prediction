import streamlit as st
import numpy as np
from joblib import load

# Load the trained model
try:
    model = load('breast_cancer_model.pkl')
    st.write("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

# Create the Streamlit app
st.title("Breast Cancer Prediction App")

# Input fields for the feature values
st.header("Enter Feature Values")

mean_radius = st.number_input("Mean Radius")
mean_texture = st.number_input("Mean Texture")
mean_perimeter = st.number_input("Mean Perimeter")
mean_area = st.number_input("Mean Area")
mean_smoothness = st.number_input("Mean Smoothness")
mean_compactness = st.number_input("Mean Compactness")
mean_concavity = st.number_input("Mean Concavity")
mean_concave_points = st.number_input("Mean Concave Points")
mean_symmetry = st.number_input("Mean Symmetry")
mean_fractal_dimension = st.number_input("Mean Fractal Dimension")

radius_error = st.number_input("Radius Error")
texture_error = st.number_input("Texture Error")
perimeter_error = st.number_input("Perimeter Error")
area_error = st.number_input("Area Error")
smoothness_error = st.number_input("Smoothness Error")
compactness_error = st.number_input("Compactness Error")
concavity_error = st.number_input("Concavity Error")
concave_points_error = st.number_input("Concave Points Error")
symmetry_error = st.number_input("Symmetry Error")
fractal_dimension_error = st.number_input("Fractal Dimension Error")

worst_radius = st.number_input("Worst Radius")
worst_texture = st.number_input("Worst Texture")
worst_perimeter = st.number_input("Worst Perimeter")
worst_area = st.number_input("Worst Area")
worst_smoothness = st.number_input("Worst Smoothness")
worst_compactness = st.number_input("Worst Compactness")
worst_concavity = st.number_input("Worst Concavity")
worst_concave_points = st.number_input("Worst Concave Points")
worst_symmetry = st.number_input("Worst Symmetry")
worst_fractal_dimension = st.number_input("Worst Fractal Dimension")

# Make a prediction using the model
if st.button("Predict"):
    if model:
        input_features = np.array([[
        mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness, mean_compactness,
        mean_concavity, mean_concave_points, mean_symmetry, mean_fractal_dimension, radius_error,
        texture_error, perimeter_error, area_error, smoothness_error, compactness_error, concavity_error,
        concave_points_error, symmetry_error, fractal_dimension_error, worst_radius, worst_texture,
        worst_perimeter, worst_area, worst_smoothness, worst_compactness, worst_concavity,
        worst_concave_points, worst_symmetry, worst_fractal_dimension
    ]])

    prediction = model.predict(input_features)

    # Display the prediction result
    st.header("Prediction Result")
    if prediction[0] == 0:
        st.success("The tumor is benign.")
    else:
        st.error("The tumor is malignant.")
else:
        st.error("Model not loaded. Cannot make predictions.")

