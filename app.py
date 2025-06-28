import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Import files
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

features = [
    'study_hours_per_day', 'part_time_job', 'attendance_percentage',
    'sleep_hours', 'diet_quality', 'exercise_frequency',
    'parental_education_level', 'internet_quality', 'mental_health_rating',
    'wasted_time'
]

categorical_cols = ['part_time_job', 'diet_quality', 'parental_education_level', 'internet_quality']
numerical_cols = [col for col in features if col not in categorical_cols]

# CSS 
st.markdown("""
    <style>
    body, .main {
        background-color: #eef4f7;
    }

    h1, h2, h3 {
        text-align: center;
        color: #1b2a41;
        font-family: 'Segoe UI', sans-serif;
    }

    .book {
        width: 300px;
        height: 200px;
        background: #ffffff;
        margin: 30px auto;
        position: relative;
        border-radius: 8px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.12);
    }

    .book::before {
        content: '';
        position: absolute;
        width: 100%;
        height: 50%;
        top: 0;
        left: 0;
        background: #d7e3f4;
        border-top-left-radius: 8px;
        border-top-right-radius: 8px;
    }

    .pen {
        width: 6px;
        height: 120px;
        background: #1b2a41;
        position: absolute;
        right: -20px;
        top: 40px;
        border-radius: 3px;
    }

    .result-sheet {
        background: #fffde7;
        border: 2px dashed #455a64;
        padding: 15px;
        margin-top: 25px;
        text-align: center;
        font-size: 22px;
        color: #000;
        border-radius: 6px;
        width: 80%;
        margin-left: auto;
        margin-right: auto;
        font-weight: bold;
        font-family: 'Segoe UI', sans-serif;
    }

    .stButton button {
        background-color: #2e7d32;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        font-size: 16px;
        padding: 10px 20px;
    }
    </style>
""", unsafe_allow_html=True)


# Like a book and pen :)
st.markdown('<div class="book"><div class="pen"></div></div>', unsafe_allow_html=True)

# Title
st.title("📚 Predict Your Exam Score")
st.markdown("Enter your daily habits and academic background to get an estimated exam score prediction.")

# User input
user_input = {}

for col in features:
    if col in categorical_cols:
        options = encoders[col].classes_
        user_input[col] = st.selectbox(f"{col.replace('_', ' ').title()}", options)
    else:
        user_input[col] = st.number_input(f"{col.replace('_', ' ').title()}", min_value=0.0)

# Predict degree
if st.button("📊 Predict"):
    input_df = pd.DataFrame([user_input])

    # Encode categorical
    for col in categorical_cols:
        input_df[col] = encoders[col].transform(input_df[col])

    # Scale numeric
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    # Predict
    prediction = model.predict(input_df)[0]
    prediction = max(0, min(prediction, 100))

    # Show result inside paper design
    st.markdown(f"<div class='result-sheet'>Predicted Exam Score: {prediction:.2f}</div>", unsafe_allow_html=True)
