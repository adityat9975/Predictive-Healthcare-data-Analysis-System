import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from pymongo import MongoClient
import random
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set up MongoDB connection
client = MongoClient("mongodb://localhost:27017")  # Adjust connection string if needed
db = client['Database']  # Database name
feedback_collection = db['Feedbacks']  # Collection for contact form feedback
medical_input_collection = db['MedicalInputs']  # Collection for storing medical inputs

# Set page configuration
st.set_page_config(page_title="Main Project", layout="wide", page_icon="??")  # Replace the icon with a valid emoji

# Load models
diabetes_model = pickle.load(open('C:/Users/Admin/Desktop/projects/project/diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open('C:/Users/Admin/Desktop/projects/project/Heart_model.sav', 'rb'))
parkinsons_model = pickle.load(open('C:/Users/Admin/Desktop/projects/project/parkinsons_model.sav', 'rb'))

# Function to validate email format
def is_valid_email(email):
    email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zAph+-]+\.[a-zA-Z]{2,}$'
    return re.match(email_regex, email) is not None

# Sidebar for navigation
with st.sidebar:
    selected = option_menu('Predictive Analysis for Multiple Disease',
                           ['Diabetes Analysis', 'Heart Disease Analysis', 'Parkinsons Analysis', 'Recommendations', 'Contact Form'],
                           menu_icon='hospital-fill', icons=['activity', 'heart', 'person', 'umbrella', 'envelope'],
                           default_index=0)

# Function to create risk classification
def classify_risk(prediction, user_input, disease_name):
    # Example threshold-based risk classification
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

# Function for displaying bar chart
def display_input_bar_chart(user_input, feature_names):
    fig, ax = plt.subplots(figsize=(8, 5))  # Adjusted figure size
    ax.bar(feature_names, user_input, color='skyblue')
    ax.set_title("User Input Data")
    ax.set_ylabel("Values")
    plt.xticks(rotation=45, ha='right', fontsize=10)  # Rotate labels and adjust font size
    st.pyplot(fig)

# Function to save medical input data into MongoDB
def save_medical_input_to_mongo(disease_name, user_input, result, risk_classification):
    medical_data = {
        "disease_name": disease_name,
        "input_data": user_input,
        "prediction_result": result,
        "risk_classification": risk_classification
    }
    medical_input_collection.insert_one(medical_data)

# Function to save contact form feedback to MongoDB
def save_feedback_to_mongo(name, email, message):
    feedback_data = {
        "name": name,
        "email": email,
        "message": message
    }
    feedback_collection.insert_one(feedback_data)

# Diabetes Prediction Page
if selected == 'Diabetes Analysis':
    st.title('Diabetes Analysis using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.number_input('Number of Pregnancies')

    with col2:
        Glucose = st.number_input('Glucose Level')

    with col3:
        BloodPressure = st.number_input('Blood Pressure value')

    with col1:
        SkinThickness = st.number_input('Skin Thickness value')

    with col2:
        Insulin = st.number_input('Insulin Level')

    with col3:
        BMI = st.number_input('BMI value')

    with col1:
        DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function value')

    with col2:
        Age = st.number_input('Age of the Person')

    if st.button('Diabetes Test Result'):
        # Prepare user input for model prediction
        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
        feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

        # Display the input data
        st.markdown("### Input Data Summary")
        input_data_df = pd.DataFrame([user_input], columns=feature_names)
        st.dataframe(input_data_df)

        # Visualize input data using bar chart
        display_input_bar_chart(user_input, feature_names)

        try:
            # Predict diabetes outcome
            diab_prediction = diabetes_model.predict([user_input])
            # Classify risk
            risk_classification = classify_risk(diab_prediction, user_input, 'diabetes')
            st.markdown(f"### Risk Classification: {risk_classification}")

            if diab_prediction[0] == 1:
                st.success('The person is diabetic')
                save_medical_input_to_mongo('diabetes', user_input, "Diabetic", risk_classification)
            else:
                st.success('The person is not diabetic')
                save_medical_input_to_mongo('diabetes', user_input, "Not Diabetic", risk_classification)
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Function to safely convert inputs to float
def safe_convert_to_float(value):
    try:
        return float(value)
    except ValueError:
        return None

# Function for displaying bar chart of input data
def display_input_bar_chart(user_input, feature_names):
    fig, ax = plt.subplots(figsize=(8, 5))  # Adjusted figure size
    ax.bar(feature_names, user_input, color='skyblue')
    ax.set_title("User Input Data")
    ax.set_ylabel("Values")
    plt.xticks(rotation=45, ha='right', fontsize=10)  # Rotate labels and adjust font size
    st.pyplot(fig)

# Heart Disease Prediction Page (Updated with Input Summary and Bar Chart)
if selected == 'Heart Disease Analysis':
    st.title('Heart Disease Analysis using ML')

    # Collecting user input
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.text_input('Age')
    with col2:
        sex = st.text_input('Gender', help='Enter 0 for male and 1 for female')
    with col3:
        cp = st.text_input('Chest Pain types')
    with col1:
        trestbps = st.text_input('Resting Blood Pressure')
    with col2:
        chol = st.text_input('Serum Cholesterol in mg/dl')
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')
    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')
    with col3:
        exang = st.text_input('Exercise Induced Angina')
    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')
    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')
    with col3:
        ca = st.text_input('Major vessels colored by fluoroscopy')
    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversible defect')

    if st.button('Heart Disease Test Result'):
        # Convert user input values to float and check if any value is invalid
        user_input = [
            safe_convert_to_float(age),
            safe_convert_to_float(sex),
            safe_convert_to_float(cp),
            safe_convert_to_float(trestbps),
            safe_convert_to_float(chol),
            safe_convert_to_float(fbs),
            safe_convert_to_float(restecg),
            safe_convert_to_float(thalach),
            safe_convert_to_float(exang),
            safe_convert_to_float(oldpeak),
            safe_convert_to_float(slope),
            safe_convert_to_float(ca),
            safe_convert_to_float(thal)
        ]

        # Check for any invalid input
        if None in user_input:
            st.error("Please enter valid numeric values for all fields.")
        else:
            try:
                # Display the input data as a table
                feature_names = [
                    'Age', 'Gender', 'Chest Pain types', 'Resting Blood Pressure', 'Serum Cholesterol',
                    'Fasting Blood Sugar', 'Resting Electrocardiographic results', 'Maximum Heart Rate',
                    'Exercise Induced Angina', 'ST depression induced by exercise', 'Slope of exercise ST segment',
                    'Major vessels colored by fluoroscopy', 'thal'
                ]
                input_data_df = pd.DataFrame([user_input], columns=feature_names)
                st.markdown("### Input Data Summary")
                st.dataframe(input_data_df)

                # Visualize input data using bar chart
                display_input_bar_chart(user_input, feature_names)

                # Predict heart disease outcome
                heart_prediction = heart_disease_model.predict([user_input])
                # Classify risk
                risk_classification = classify_risk(heart_prediction, user_input, 'heart_disease')
                st.markdown(f"### Risk Classification: {risk_classification}")

                if heart_prediction[0] == 1:
                    st.success('The person is having heart disease')
                    save_medical_input_to_mongo('heart_disease', user_input, "Heart Disease", risk_classification)
                else:
                    st.success('The person does not have any heart disease')
                    save_medical_input_to_mongo('heart_disease', user_input, "No Heart Disease", risk_classification)

            except Exception as e:
                st.error(f"An error occurred: {e}")

# Parkinson's Disease Prediction Page (Updated to include all 22 features, input summary, and bar chart)
if selected == "Parkinsons Analysis":
    st.title("Parkinson's Disease Analysis using ML")

    # Define columns for organizing inputs
    col1, col2, col3, col4, col5 = st.columns(5)

    # Collecting all 22 features for input
    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')
    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')
    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')
    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')
    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
    
    with col1:
        RAP = st.text_input('MDVP:RAP')
    with col2:
        PPQ = st.text_input('MDVP:PPQ')
    with col3:
        DDP = st.text_input('Jitter:DDP')
    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')
    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
    
    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')
    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')
    with col3:
        APQ = st.text_input('MDVP:APQ')
    with col4:
        DDA = st.text_input('Shimmer:DDA')
    with col5:
        NHR = st.text_input('NHR')
    
    with col1:
        HNR = st.text_input('HNR')
    with col2:
        RPDE = st.text_input('RPDE')
    with col3:
        DFA = st.text_input('DFA')
    with col4:
        spread1 = st.text_input('spread1')
    with col5:
        spread2 = st.text_input('spread2')
    
    with col1:
        D2 = st.text_input('D2')
    with col2:
        PPE = st.text_input('PPE')

    if st.button("Parkinson's Test Result"):
        # Convert user input values to float and check if any value is invalid
        user_input = [
            safe_convert_to_float(fo),
            safe_convert_to_float(fhi),
            safe_convert_to_float(flo),
            safe_convert_to_float(Jitter_percent),
            safe_convert_to_float(Jitter_Abs),
            safe_convert_to_float(RAP),
            safe_convert_to_float(PPQ),
            safe_convert_to_float(DDP),
            safe_convert_to_float(Shimmer),
            safe_convert_to_float(Shimmer_dB),
            safe_convert_to_float(APQ3),
            safe_convert_to_float(APQ5),
            safe_convert_to_float(APQ),
            safe_convert_to_float(DDA),
            safe_convert_to_float(NHR),
            safe_convert_to_float(HNR),
            safe_convert_to_float(RPDE),
            safe_convert_to_float(DFA),
            safe_convert_to_float(spread1),
            safe_convert_to_float(spread2),
            safe_convert_to_float(D2),
            safe_convert_to_float(PPE)
        ]

        # Check for any invalid input
        if None in user_input:
            st.error("Please enter valid numeric values for all fields.")
        else:
            try:
                # Display the input data as a table
                feature_names = [
                    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)',
                    'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 
                    'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 
                    'D2', 'PPE'
                ]
                input_data_df = pd.DataFrame([user_input], columns=feature_names)
                st.markdown("### Input Data Summary")
                st.dataframe(input_data_df)

                # Visualize input data using bar chart
                display_input_bar_chart(user_input, feature_names)

                # Predict Parkinson's disease outcome
                parkinsons_prediction = parkinsons_model.predict([user_input])
                # Classify risk
                risk_classification = classify_risk(parkinsons_prediction, user_input, 'parkinsons')
                st.markdown(f"### Risk Classification: {risk_classification}")

                if parkinsons_prediction[0] == 1:
                    st.success("The person has Parkinson's disease")
                    save_medical_input_to_mongo('parkinsons', user_input, "Parkinson's Disease", risk_classification)
                else:
                    st.success("The person does not have Parkinson's disease")
                    save_medical_input_to_mongo('parkinsons', user_input, "No Parkinson's Disease", risk_classification)

            except Exception as e:
                st.error(f"An error occurred: {e}")









if selected == 'Recommendations':
    st.title('Diet and Exercise Recommendations')

    # Option menu to select disease
    disease = st.selectbox("Select Disease", ["Diabetes", "Heart Disease", "Parkinson's Disease"])

    # Select age and weight from dropdowns
    age = st.selectbox("Select Age (years)", list(range(10, 101)))  # Age dropdown from 10 to 100
    weight = st.selectbox("Select Weight (kg)", list(range(30, 151)))  # Weight dropdown from 30kg to 150kg

    # Select risk level (pre-calculated)
    risk_level = st.selectbox("Select Risk Level", ["Low", "Moderate", "High"])

    # Select activity level
    activity_level = st.selectbox("Select Activity Level", ["Sedentary", "Lightly Active", "Moderately Active", "Very Active"])

    # Select body fat percentage
    body_fat_percentage = st.selectbox("Select Body Fat Percentage", ["Low", "Normal", "High"])

    # Gender selection (affects caloric intake)
    gender = st.selectbox("Select Gender", ["Male", "Female"])

    # Display Disease, Risk Level, Age, Weight, and Activity Level
    st.write(f"Selected Disease: {disease}")
    st.write(f"Age: {age} years")
    st.write(f"Weight: {weight} kg")
    st.write(f"Risk Level: {risk_level}")
    st.write(f"Activity Level: {activity_level}")
    st.write(f"Body Fat Percentage: {body_fat_percentage}")
    st.write(f"Gender: {gender}")

    # Calculate Recommended Caloric Intake Based on Age, Weight, Risk Level, and Gender
    def calculate_calories(age, weight, risk_level, gender):
        # Basic formula: Estimated Caloric Requirement (Mifflin-St Jeor Equation)
        if gender == "Male":
            if age < 18:
                bmr = 17.5 * weight + 651  # for teens, adjust as per formula
            else:
                bmr = 10 * weight + 6.25 * age - 5 * age + 5  # Mifflin-St Jeor for men
        else:  # Female
            if age < 18:
                bmr = 14.7 * weight + 496  # for teens, adjust as per formula
            else:
                bmr = 10 * weight + 6.25 * age - 5 * age - 161  # Mifflin-St Jeor for women

        # Adjust caloric intake based on risk level
        if risk_level == "High":
            caloric_intake = bmr * 1.4  # Active lifestyle with high disease risk
        elif risk_level == "Moderate":
            caloric_intake = bmr * 1.2  # Moderate activity with moderate disease risk
        else:
            caloric_intake = bmr * 1.1  # Lower activity, lower disease risk

        return caloric_intake

    recommended_calories = calculate_calories(age, weight, risk_level, gender)
    st.write(f"Recommended Daily Caloric Intake: {recommended_calories:.0f} kcal")

    # Adjust calories based on activity level
    def adjust_calories_for_activity(calories, activity_level):
        if activity_level == "Sedentary":
            return calories * 1.2
        elif activity_level == "Lightly Active":
            return calories * 1.375
        elif activity_level == "Moderately Active":
            return calories * 1.55
        elif activity_level == "Very Active":
            return calories * 1.725

    adjusted_calories = adjust_calories_for_activity(recommended_calories, activity_level)
    st.write(f"Adjusted Daily Caloric Intake based on Activity Level: {adjusted_calories:.0f} kcal")

    # Function to provide recommendations based on disease, risk level, and calories
    def provide_recommendations(disease, risk_level, age, weight, calories):
        # Diet Recommendations
        if disease == "Diabetes":
            if risk_level == "High":
                st.write("### Diet Recommendations:")
                st.write("- **Low-carb, high-fiber diet**: Limit sugar and refined carbs. Focus on leafy greens, whole grains, and legumes.")
                st.write("- **Healthy fats**: Incorporate healthy fats such as avocado, olive oil, and omega-3-rich foods like salmon.")
                if calories > 1500:
                    st.write("- **Consider reducing calorie intake**: Aiming for a calorie intake of around 1500-1600 kcal/day can help manage blood sugar levels.")
            elif risk_level == "Moderate":
                st.write("### Diet Recommendations:")
                st.write("- **Moderate-carb, low-sugar diet**: Include whole grains, lean proteins like chicken and fish, and plenty of vegetables.")
                st.write("- **Increase fiber intake**: Foods like oats, quinoa, and beans can be beneficial.")
            else:
                st.write("### Diet Recommendations:")
                st.write("- **Balanced diet**: Focus on low glycemic index foods like quinoa, sweet potatoes, and leafy greens.")
            
            # Foods to avoid for Diabetes
            st.write("### Foods to Avoid for Diabetes:")
            st.write("- **Sugary drinks**: Soda, sweetened tea, and sugary coffee drinks can spike blood sugar levels.")
            st.write("- **Refined carbs**: White bread, pasta, and rice can lead to increased blood sugar.")
            st.write("- **Processed snacks**: Potato chips, cookies, and pastries often contain unhealthy fats and refined sugars.")
            
            # Instructions to reduce risk of Diabetes
            st.write("### Instructions to Reduce the Risk of Diabetes:")
            st.write("- **Exercise regularly**: Aim for at least 30 minutes of moderate activity daily, like walking or swimming.")
            st.write("- **Maintain a healthy weight**: Try to stay at a healthy weight to manage blood sugar levels effectively.")
            st.write("- **Eat more fiber**: Fiber-rich foods like vegetables, fruits, and whole grains can help regulate blood sugar.")
            st.write("- **Monitor blood sugar**: Regularly check your blood sugar levels, especially if you’re pre-diabetic.")

        elif disease == "Heart Disease":
            if risk_level == "High":
                st.write("### Diet Recommendations:")
                st.write("- **Low-fat, heart-healthy diet**: Avoid trans fats, processed foods, and limit sodium intake. Focus on whole grains and vegetables.")
                st.write("- **Include more fish**: Salmon, mackerel, and other fatty fish rich in omega-3s.")
                if calories > 1800:
                    st.write("- **Consider reducing calorie intake**: Aim for a calorie intake of around 1500-1800 kcal/day to manage heart health.")
            elif risk_level == "Moderate":
                st.write("### Diet Recommendations:")
                st.write("- **Low-sodium, heart-healthy diet**: Include more fish, vegetables, and whole grains.")
            else:
                st.write("### Diet Recommendations:")
                st.write("- **Balanced diet**: Emphasize whole grains, vegetables, healthy fats, and lean proteins.")

            # Foods to avoid for Heart Disease
            st.write("### Foods to Avoid for Heart Disease:")
            st.write("- **Trans fats**: Found in many processed and packaged foods, trans fats increase bad cholesterol.")
            st.write("- **High-sodium foods**: Salted snacks, canned soups, and fast food can contribute to high blood pressure.")
            st.write("- **Saturated fats**: Found in fatty cuts of meat and full-fat dairy, which can raise cholesterol levels.")
            
            # Instructions to reduce risk of Heart Disease
            st.write("### Instructions to Reduce the Risk of Heart Disease:")
            st.write("- **Regular physical activity**: Engage in cardiovascular exercises such as walking, running, or cycling.")
            st.write("- **Manage stress**: Practice stress-reducing techniques like yoga, meditation, or deep breathing.")
            st.write("- **Avoid smoking and excessive alcohol consumption**: These are major risk factors for heart disease.")
            st.write("- **Get enough sleep**: Aim for 7-9 hours of sleep per night to help your heart function properly.")

        elif disease == "Parkinson's Disease":
            if risk_level == "High":
                st.write("### Diet Recommendations:")
                st.write("- **Anti-inflammatory, nutrient-dense foods**: Focus on omega-3s from fatty fish, and antioxidants from vegetables like spinach and kale.")
                st.write("- **Smaller meals**: Consider smaller, more frequent meals to manage appetite and avoid overeating.")
                if calories < 2000:
                    st.write("- **Increase calorie intake**: Consider aiming for 2000-2500 kcal/day to prevent weight loss and muscle wasting.")
            elif risk_level == "Moderate":
                st.write("### Diet Recommendations:")
                st.write("- **Balanced diet**: Include plenty of fruits, vegetables, lean proteins, and whole grains.")
            else:
                st.write("### Diet Recommendations:")
                st.write("- **Maintain healthy diet**: Incorporate lots of vegetables, fruits, and lean proteins.")

    # Exercise Recommendations
    if disease == "Diabetes":
        st.write("### Exercise Recommendations:")
        if risk_level == "High":
            st.write("- **High-intensity interval training (HIIT)**: Incorporate short bursts of intense exercise followed by rest periods. It can help with blood sugar control.")
            st.write("- **Strength training**: Focus on weight-bearing exercises at least twice a week to improve muscle mass and glucose metabolism.")
            st.write("- **Aerobic exercise**: Engage in moderate-intensity aerobic exercise, like brisk walking or cycling, for at least 30 minutes daily.")
        elif risk_level == "Moderate":
            st.write("- **Moderate aerobic exercise**: Aim for activities like walking, swimming, or biking for 30 minutes most days of the week.")
            st.write("- **Strength training**: Incorporate light weights or resistance bands 2-3 times a week.")
        else:
            st.write("- **Low to moderate exercise**: Gentle activities like walking or yoga can be beneficial to help with glucose regulation.")

    elif disease == "Heart Disease":
        st.write("### Exercise Recommendations:")
        if risk_level == "High":
            st.write("- **Cardiovascular exercise**: Engage in aerobic activities like walking, cycling, or swimming for at least 30 minutes, 5 days a week.")
            st.write("- **Strength training**: Perform moderate strength exercises twice a week to improve overall strength.")
            st.write("- **Low-impact activities**: Walking, water aerobics, and cycling can reduce stress on the heart.")
        elif risk_level == "Moderate":
            st.write("- **Moderate aerobic exercise**: 150 minutes of walking or cycling per week.")
            st.write("- **Strength training**: Light strength exercises twice a week.")
        else:
            st.write("- **Mild exercise**: Gentle walking, stretching, or low-intensity yoga can help with maintaining heart health.")

    elif disease == "Parkinson's Disease":
        st.write("### Exercise Recommendations:")
        if risk_level == "High":
            st.write("- **Balance and coordination exercises**: Activities like Tai Chi, yoga, or dancing can help improve balance and reduce falls.")
            st.write("- **Strength training**: Focus on building strength to maintain muscle mass. Aim for 2-3 times a week with resistance exercises.")
            st.write("- **Aerobic exercise**: Moderate activities like walking, swimming, or stationary biking for 30 minutes daily can improve mobility and energy levels.")
        elif risk_level == "Moderate":
            st.write("- **Low-impact aerobic exercise**: Walking or cycling for 20-30 minutes daily.")
            st.write("- **Stretching**: Gentle stretching exercises to improve flexibility.")
        else:
            st.write("- **Light exercise**: Gentle activities like walking, stretching, or seated exercises can help manage symptoms.")

    # Further recommendations based on diet, activity level, etc...

            # Foods to avoid for Parkinson's Disease
            st.write("### Foods to Avoid for Parkinson's Disease:")
            st.write("- **High-fat, processed foods**: These can increase inflammation and worsen symptoms.")
            st.write("- **Excessive dairy**: Some studies suggest a potential link between high dairy consumption and Parkinson's.")
            st.write("- **Caffeine and stimulants**: Too much caffeine can interfere with Parkinson's medications.")

            # Instructions to reduce risk of Parkinson's Disease
            st.write("### Instructions to Reduce the Risk of Parkinson's Disease:")
            st.write("- **Exercise regularly**: Engage in regular physical activity, including strength training and balance exercises.")
            st.write("- **Get enough sleep**: Adequate rest is essential for managing Parkinson's symptoms.")
            st.write("- **Consider a Mediterranean diet**: This type of diet, rich in fruits, vegetables, and healthy fats, may reduce inflammation.")
            st.write("- **Mental stimulation**: Engage in activities like puzzles or reading to keep your brain active and healthy.")

    # Provide recommendations based on the inputs
    provide_recommendations(disease, risk_level, age, weight, adjusted_calories)

# Contact Form Page
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

