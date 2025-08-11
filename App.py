import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import random
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Optional MongoDB connection (only if MONGO_URI is set in environment variables)
try:
    from pymongo import MongoClient
    mongo_uri = os.getenv("MONGO_URI", None)
    if mongo_uri:
        client = MongoClient(mongo_uri)
        db = client['Database']
        feedback_collection = db['Feedbacks']
        medical_input_collection = db['MedicalInputs']
    else:
        client = None
        feedback_collection = None
        medical_input_collection = None
except Exception:
    client = None
    feedback_collection = None
    medical_input_collection = None

# Set page configuration
st.set_page_config(page_title="Predictive Healthcare Data Analysis", layout="wide", page_icon="💊")

# Load models from local 'models' folder
MODEL_DIR = "model"
diabetes_model = pickle.load(open(os.path.join(MODEL_DIR, 'diabetes_model.sav'), 'rb'))
heart_disease_model = pickle.load(open(os.path.join(MODEL_DIR, 'Heart_model.sav'), 'rb'))
parkinsons_model = pickle.load(open(os.path.join(MODEL_DIR, 'parkinsons_model.sav'), 'rb'))

# Email validation function
def is_valid_email(email):
    email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(email_regex, email) is not None

# Sidebar menu
with st.sidebar:
    selected = option_menu(
        'Predictive Analysis for Multiple Disease',
        ['Diabetes Analysis', 'Heart Disease Analysis', 'Parkinsons Analysis', 'Recommendations', 'Contact Form'],
        menu_icon='hospital-fill',
        icons=['activity', 'heart', 'person', 'umbrella', 'envelope'],
        default_index=0
    )

# Risk classification function
def classify_risk(prediction, user_input, disease_name):
    if disease_name == 'diabetes':
        risk_score = sum(user_input) / len(user_input)
        if risk_score < 0.4:
            return "Low Risk"
        elif 0.4 <= risk_score < 0.7:
            return "Moderate Risk"
        else:
            return "High Risk"
    elif disease_name == 'heart_disease':
        return "Moderate Risk" if prediction[0] == 1 else "Low Risk"
    elif disease_name == 'parkinsons':
        return "High Risk" if prediction[0] == 1 else "Low Risk"

# Display bar chart
def display_input_bar_chart(user_input, feature_names):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(feature_names, user_input, color='skyblue')
    ax.set_title("User Input Data")
    ax.set_ylabel("Values")
    plt.xticks(rotation=45, ha='right', fontsize=10)
    st.pyplot(fig)

# Safe DB save functions
def save_medical_input_to_mongo(disease_name, user_input, result, risk_classification):
    if medical_input_collection:
        medical_input_collection.insert_one({
            "disease_name": disease_name,
            "input_data": user_input,
            "prediction_result": result,
            "risk_classification": risk_classification
        })

def save_feedback_to_mongo(name, email, message):
    if feedback_collection:
        feedback_collection.insert_one({
            "name": name,
            "email": email,
            "message": message
        })

# Diabetes Analysis
if selected == 'Diabetes Analysis':
    st.title('Diabetes Analysis using ML')
    cols = st.columns(3)
    Pregnancies = cols[0].number_input('Number of Pregnancies')
    Glucose = cols[1].number_input('Glucose Level')
    BloodPressure = cols[2].number_input('Blood Pressure value')
    SkinThickness = cols[0].number_input('Skin Thickness value')
    Insulin = cols[1].number_input('Insulin Level')
    BMI = cols[2].number_input('BMI value')
    DiabetesPedigreeFunction = cols[0].number_input('Diabetes Pedigree Function value')
    Age = cols[1].number_input('Age of the Person')

    if st.button('Diabetes Test Result'):
        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
        feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        st.markdown("### Input Data Summary")
        st.dataframe(pd.DataFrame([user_input], columns=feature_names))
        display_input_bar_chart(user_input, feature_names)
        prediction = diabetes_model.predict([user_input])
        risk = classify_risk(prediction, user_input, 'diabetes')
        st.markdown(f"### Risk Classification: {risk}")
        result_text = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
        st.success(f"The person is {result_text}")
        save_medical_input_to_mongo('diabetes', user_input, result_text, risk)

# Heart Disease Analysis
if selected == 'Heart Disease Analysis':
    st.title('Heart Disease Analysis using ML')
    feature_names = ['Age', 'Gender', 'Chest Pain types', 'Resting Blood Pressure', 'Serum Cholesterol',
                     'Fasting Blood Sugar', 'Resting Electrocardiographic results', 'Maximum Heart Rate',
                     'Exercise Induced Angina', 'ST depression induced by exercise', 'Slope of exercise ST segment',
                     'Major vessels colored by fluoroscopy', 'thal']
    user_input = [st.text_input(name) for name in feature_names]

    if st.button('Heart Disease Test Result'):
        try:
            user_input = [float(x) for x in user_input]
            st.markdown("### Input Data Summary")
            st.dataframe(pd.DataFrame([user_input], columns=feature_names))
            display_input_bar_chart(user_input, feature_names)
            prediction = heart_disease_model.predict([user_input])
            risk = classify_risk(prediction, user_input, 'heart_disease')
            st.markdown(f"### Risk Classification: {risk}")
            result_text = "Heart Disease" if prediction[0] == 1 else "No Heart Disease"
            st.success(f"The person has {result_text}")
            save_medical_input_to_mongo('heart_disease', user_input, result_text, risk)
        except:
            st.error("Please enter valid numeric values.")

# Parkinson's Disease Analysis
if selected == "Parkinsons Analysis":
    st.title("Parkinson's Disease Analysis using ML")
    feature_names = [
        'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)',
        'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 
        'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 
        'D2', 'PPE'
    ]
    user_input = [st.text_input(name) for name in feature_names]

    if st.button("Parkinson's Test Result"):
        try:
            user_input = [float(x) for x in user_input]
            st.markdown("### Input Data Summary")
            st.dataframe(pd.DataFrame([user_input], columns=feature_names))
            display_input_bar_chart(user_input, feature_names)
            prediction = parkinsons_model.predict([user_input])
            risk = classify_risk(prediction, user_input, 'parkinsons')
            st.markdown(f"### Risk Classification: {risk}")
            result_text = "Parkinson's Disease" if prediction[0] == 1 else "No Parkinson's Disease"
            st.success(f"The person has {result_text}")
            save_medical_input_to_mongo('parkinsons', user_input, result_text, risk)
        except:
            st.error("Please enter valid numeric values.")

# Contact Form
if selected == 'Contact Form':
    st.title('Contact Us')
    with st.form("contact_form"):
        name = st.text_input("Your Name")
        email = st.text_input("Your Email")
        message = st.text_area("Your Message")
        submit_button = st.form_submit_button("Submit")

    if submit_button:
        if not is_valid_email(email):
            st.error("Please enter a valid email address.")
        else:
            save_feedback_to_mongo(name, email, message)
            st.success("Your message has been sent successfully!")
