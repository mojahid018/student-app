import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Function to load the model
def load_model():
    model_path = "linear_regression_model.pkl"  # Ensure this file exists
    if not os.path.exists(model_path):
        st.error("Model file not found! Ensure 'linear_regression_model.pkl' exists in the same directory.")
        return None, None, None
    with open(model_path, "rb") as file:
        model, scaler, le = pickle.load(file)
    return model, scaler, le

# Function to preprocess input data
def preprocessing_input_data(data, scaler, le):
    data["Extracurricular Activities"] = le.transform([data["Extracurricular Activities"]])[0]  # Encode categorical value
    df = pd.DataFrame([data])  # Convert to DataFrame
    df_transformed = scaler.transform(df)  # Apply scaling
    return df_transformed

# Function to make predictions
def predict_data(data):
    model, scaler, le = load_model()
    if model is None:
        return "Error: Model not loaded."
    preprocessed_data = preprocessing_input_data(data, scaler, le)
    prediction = model.predict(preprocessed_data)
    return round(prediction[0], 2)

# Main function for Streamlit UI
def main():
    st.title("Student Performance Prediction")
    st.write("Enter your details to get a prediction on your score.")

    # User input fields
    hour_studied = st.number_input("Hours Studied", min_value=1, max_value=10, value=5)
    previous_scores = st.number_input("Previous Scores", min_value=40, max_value=100, value=70)
    extracurricular_activities = st.selectbox("Extracurricular Activities", ["Yes", "No"])
    sleep_hours = st.number_input("Sleep Hours", min_value=4, max_value=10, value=7)
    question_papers = st.number_input("Sample Question Papers Practiced", min_value=0, max_value=10, value=5)

    if st.button("Predict Your Score"):
        user_data = {
            "Hours Studied": hour_studied,
            "Previous Scores": previous_scores,
            "Extracurricular Activities": extracurricular_activities,
            "Sleep Hours": sleep_hours,
            "Sample Question Papers Practiced": question_papers,
        }
        prediction = predict_data(user_data)
        if isinstance(prediction, str):  # Handle error case
            st.error(prediction)
        else:
            st.success(f"Your predicted score is: {prediction}")

if __name__ == "__main__":
    main()

