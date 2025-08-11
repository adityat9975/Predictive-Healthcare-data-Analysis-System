import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from streamlit_extras.let_it_rain import rain
from PIL import Image

# Optional MongoDB connection (only if MONGO_URI is set)
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

# Page config
st.set_page_config(
    page_title="Predictive Healthcare Analysis",
    layout="wide",
    page_icon="💊",
    initial_sidebar_state="expanded"
)

# Load models from local 'model' folder
MODEL_DIR = "model"
try:
    diabetes_model = pickle.load(open(os.path.join(MODEL_DIR, 'diabetes_model.sav'), 'rb'))
    heart_disease_model = pickle.load(open(os.path.join(MODEL_DIR, 'Heart_model.sav'), 'rb'))
    parkinsons_model = pickle.load(open(os.path.join(MODEL_DIR, 'parkinsons_model.sav'), 'rb'))
except FileNotFoundError:
    st.error("Model files not found. Please ensure 'diabetes_model.sav', 'Heart_model.sav', and 'parkinsons_model.sav' are in the 'model' directory.")
    st.stop()


# Email validation function
def is_valid_email(email):
    email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(email_regex, email) is not None

# Display bar chart
def display_input_bar_chart(user_input, feature_names):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(feature_names, user_input, color='teal')
    ax.set_title("User Input Data", fontsize=16)
    ax.set_ylabel("Values", fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)

# Risk classification function
def classify_risk(prediction, user_input, disease_name):
    if disease_name == 'diabetes':
        risk_score = (sum(user_input) / len(user_input)) / 100 # Normalize the score
        if risk_score < 0.2:
            return "Low Risk"
        elif 0.2 <= risk_score < 0.5:
            return "Moderate Risk"
        else:
            return "High Risk"
    elif disease_name == 'heart_disease':
        return "High Risk" if prediction[0] == 1 else "Low Risk"
    elif disease_name == 'parkinsons':
        return "High Risk" if prediction[0] == 1 else "Low Risk"

# Safe DB save functions
def save_medical_input_to_mongo(disease_name, user_input, result, risk_classification):
    if medical_input_collection:
        try:
            medical_input_collection.insert_one({
                "disease_name": disease_name,
                "input_data": user_input,
                "prediction_result": result,
                "risk_classification": risk_classification
            })
        except Exception as e:
            st.error(f"Failed to save data to MongoDB: {e}")

def save_feedback_to_mongo(name, email, message):
    if feedback_collection:
        try:
            feedback_collection.insert_one({
                "name": name,
                "email": email,
                "message": message
            })
        except Exception as e:
            st.error(f"Failed to save feedback to MongoDB: {e}")

# Sidebar menu
with st.sidebar:
    st.image("https://i.imgur.com/R3t6f4Q.png", use_column_width=True) # Add a logo or image
    st.markdown("<h1 style='text-align: center;'>Predictive Health Analyzer</h1>", unsafe_allow_html=True)
    selected = option_menu(
        'Main Menu',
        ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction', 'Health Recommendations', 'Contact & Feedback'],
        menu_icon='hospital-fill',
        icons=['activity', 'heart-pulse', 'person-walking', 'clipboard2-pulse', 'envelope'],
        default_index=0
    )

### 📌 Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    st.title('🩺 Diabetes Prediction')
    st.markdown("---")
    st.info("Please enter the following details to predict the likelihood of diabetes.")
    
    with st.container(border=True):
        st.subheader("Patient Vitals")
        col1, col2 = st.columns(2)
        with col1:
            Pregnancies = st.slider('Number of Pregnancies', 0, 17, 3, help="Number of times pregnant")
            Glucose = st.number_input('Glucose Level (mg/dL)', min_value=0, max_value=200, value=120)
            BloodPressure = st.number_input('Blood Pressure (mmHg)', min_value=0, max_value=122, value=70)
        with col2:
            SkinThickness = st.number_input('Skin Thickness (mm)', min_value=0, max_value=99, value=20)
            Insulin = st.number_input('Insulin Level (mu U/ml)', min_value=0, max_value=846, value=79)
            BMI = st.number_input('BMI value', min_value=0.0, max_value=67.1, value=32.0)
        
        col3, col4 = st.columns(2)
        with col3:
            DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function value', min_value=0.0, max_value=2.42, value=0.47)
        with col4:
            Age = st.slider('Age of the Person', 1, 100, 30)

    diabetes_diagnosis = ''
    if st.button('Predict Diabetes Status', use_container_width=True, type="primary"):
        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
        feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DPF', 'Age']
        
        with st.spinner('Analyzing data...'):
            prediction = diabetes_model.predict([user_input])
            risk = classify_risk(prediction, user_input, 'diabetes')
        
        st.divider()
        st.subheader("Prediction Results")
        
        cols_res = st.columns(2)
        with cols_res[0]:
            if prediction[0] == 1:
                st.error("The person is likely to be **Diabetic**.")
                st.metric(label="Risk Level", value=risk, delta_color="inverse")
                st.warning("It's highly recommended to consult a doctor and follow their advice.")
            else:
                st.success("The person is likely **Not Diabetic**.")
                st.metric(label="Risk Level", value=risk, delta_color="normal")
                st.info("Keep up the good work on maintaining a healthy lifestyle.")
        with cols_res[1]:
            st.markdown("### Input Data Visualization")
            display_input_bar_chart(user_input, feature_names)
        
        save_medical_input_to_mongo('diabetes', user_input, "Diabetic" if prediction[0] == 1 else "Not Diabetic", risk)

### 💖 Heart Disease Prediction Page
elif selected == 'Heart Disease Prediction':
    st.title('❤ Heart Disease Prediction')
    st.markdown("---")
    st.info("Fill in the details below to predict your heart health.")

    with st.container(border=True):
        st.subheader("Patient Information")
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider('Age', 1, 120, 50, help="Age in years")
            sex = st.selectbox('Gender', options=['Male', 'Female'], index=0)
            sex_val = 1 if sex == 'Male' else 0
            cp = st.selectbox('Chest Pain Type', options=['Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asymptomatic'], index=0, help="Chest pain experience")
            cp_val = ['typical angina', 'atypical angina', 'non-anginal pain', 'asymptomatic'].index(cp.lower())
            trestbps = st.slider('Resting Blood Pressure (mmHg)', 94, 200, 120)
            chol = st.slider('Serum Cholesterol (mg/dl)', 126, 564, 240)
            fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', options=['No', 'Yes'], index=0)
            fbs_val = 1 if fbs == 'Yes' else 0
        with col2:
            restecg = st.selectbox('Resting Electrocardiographic Results', options=['Normal', 'ST-T wave abnormality', 'Left ventricular hypertrophy'], index=0)
            restecg_val = ['normal', 'st-t wave abnormality', 'left ventricular hypertrophy'].index(restecg.lower())
            thalach = st.slider('Maximum Heart Rate Achieved', 71, 202, 150)
            exang = st.selectbox('Exercise Induced Angina', options=['No', 'Yes'], index=0)
            exang_val = 1 if exang == 'Yes' else 0
            oldpeak = st.slider('ST Depression Induced by Exercise', 0.0, 6.2, 1.0)
            slope = st.selectbox('Slope of the Peak Exercise ST Segment', options=['Upsloping', 'Flat', 'Downsloping'], index=0)
            slope_val = ['upsloping', 'flat', 'downsloping'].index(slope.lower()) + 1
            ca = st.slider('Number of Major Vessels Colored by Fluoroscopy', 0, 4, 0)
            thal = st.selectbox('Thalassemia', options=['Normal', 'Fixed Defect', 'Reversable Defect'], index=0)
            thal_val = ['normal', 'fixed defect', 'reversable defect'].index(thal.lower()) + 1

    heart_disease_diagnosis = ''
    if st.button('Predict Heart Disease', use_container_width=True, type="primary"):
        user_input = [age, sex_val, cp_val, trestbps, chol, fbs_val, restecg_val, thalach, exang_val, oldpeak, slope_val, ca, thal_val]
        feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        
        with st.spinner('Analyzing data...'):
            prediction = heart_disease_model.predict([user_input])
            risk = classify_risk(prediction, user_input, 'heart_disease')
        
        st.divider()
        st.subheader("Prediction Results")
        
        cols_res = st.columns(2)
        with cols_res[0]:
            if prediction[0] == 1:
                st.error("The person is likely to have **Heart Disease**.")
                st.metric(label="Risk Level", value=risk, delta_color="inverse")
                st.warning("Immediate medical consultation is strongly advised.")
            else:
                st.success("The person is likely **Not to have Heart Disease**.")
                st.metric(label="Risk Level", value=risk, delta_color="normal")
                st.info("Continue with your healthy lifestyle and regular check-ups.")
        with cols_res[1]:
            st.markdown("### Input Data Visualization")
            display_input_bar_chart(user_input, feature_names)
        
        save_medical_input_to_mongo('heart_disease', user_input, "Heart Disease" if prediction[0] == 1 else "No Heart Disease", risk)

### 🚶 Parkinsons Prediction Page
elif selected == "Parkinsons Prediction":
    st.title("🚶 Parkinson's Disease Prediction")
    st.markdown("---")
    st.info("Input the vocal parameters to check for Parkinson's disease.")
    
    with st.container(border=True):
        st.subheader("Vocal Metrics")
        cols = st.columns(4)
        feature_names = [
            'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)',
            'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 
            'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 
            'D2', 'PPE'
        ]
        user_input = []
        for i, name in enumerate(feature_names):
            user_input.append(cols[i % 4].text_input(name, key=f"parkinson_input_{i}"))

    parkinsons_diagnosis = ''
    if st.button("Predict Parkinson's Status", use_container_width=True, type="primary"):
        try:
            user_input = [float(x) for x in user_input]
            with st.spinner('Analyzing data...'):
                prediction = parkinsons_model.predict([user_input])
                risk = classify_risk(prediction, user_input, 'parkinsons')
            
            st.divider()
            st.subheader("Prediction Results")
            
            cols_res = st.columns(2)
            with cols_res[0]:
                if prediction[0] == 1:
                    st.error("The person is likely to have **Parkinson's Disease**.")
                    st.metric(label="Risk Level", value=risk, delta_color="inverse")
                    st.warning("Please consult a specialist for further examination.")
                else:
                    st.success("The person is likely **Not to have Parkinson's Disease**.")
                    st.metric(label="Risk Level", value=risk, delta_color="normal")
                    st.info("Maintain vocal health with regular exercises.")
            with cols_res[1]:
                st.markdown("### Input Data Visualization")
                display_input_bar_chart(user_input, feature_names)

            save_medical_input_to_mongo('parkinsons', user_input, "Parkinson's Disease" if prediction[0] == 1 else "No Parkinson's Disease", risk)
        except ValueError:
            st.error("Please ensure all inputs are valid numeric values.")

### ☂️ Recommendations Page
elif selected == "Health Recommendations":
    st.title("🌿 Personalized Health Recommendations")
    st.markdown("---")
    st.subheader("General Wellness Tips")
    st.markdown("""
    - **Maintain a Balanced Diet:** 🍎 Eat a variety of fruits, vegetables, and whole grains. Reduce intake of processed foods and sugary drinks.
    - **Regular Physical Activity:** 🏃 Engage in at least 30 minutes of moderate exercise most days of the week.
    - **Prioritize Sleep:** 😴 Aim for 7-9 hours of quality sleep per night to allow your body to recover.
    - **Manage Stress:** 🧘 Practice mindfulness, meditation, or yoga to help reduce stress levels.
    - **Stay Hydrated:** 💧 Drink plenty of water throughout the day.
    """)
    
    st.subheader("Tips Based on Your Risk Factors")
    st.markdown("""
    Based on your analysis, here are some tailored recommendations:
    - **For Diabetes Risk:** Focus on controlling your blood sugar levels. Increase fiber intake and monitor carbohydrate consumption.
    - **For Heart Disease Risk:** Adopt a heart-healthy diet low in saturated fats and cholesterol. Limit sodium and alcohol intake.
    - **For Parkinson's Risk:** Engage in vocal exercises to maintain voice strength. Physical therapy and regular movement can help with motor symptoms.
    """)
    st.info("Disclaimer: These are general recommendations. Always consult a healthcare professional for personalized medical advice.")

### 📬 Contact & Feedback Page
elif selected == 'Contact & Feedback':
    st.title('✉ Contact Us')
    st.markdown("---")
    st.info("We'd love to hear your thoughts and feedback!")
    
    with st.form("contact_form", clear_on_submit=True):
        name = st.text_input("Your Name", placeholder="e.g., John Doe")
        email = st.text_input("Your Email", placeholder="e.g., john.doe@example.com")
        message = st.text_area("Your Message", placeholder="Type your message here...")
        
        submit_button = st.form_submit_button("Submit Feedback", type="primary")

    if submit_button:
        if not name or not email or not message:
            st.error("Please fill in all fields.")
        elif not is_valid_email(email):
            st.error("Please enter a valid email address.")
        else:
            save_feedback_to_mongo(name, email, message)
            st.success("Thank you for your feedback! Your message has been sent successfully.")
            rain(emoji="👍", font_size=54, falling_speed=5, animation_length="100")
