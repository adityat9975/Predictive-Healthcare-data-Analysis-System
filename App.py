import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import hashlib
import json
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional MongoDB connection (only if MONGO_URI is set)
try:
    from pymongo import MongoClient
    mongo_uri = os.getenv("MONGO_URI", None)
    if mongo_uri:
        client = MongoClient(mongo_uri)
        db = client['HealthcareDB']
        feedback_collection = db['Feedbacks']
        medical_input_collection = db['MedicalInputs']
        users_collection = db['Users']
        reports_collection = db['Reports']
    else:
        client = None
        feedback_collection = None
        medical_input_collection = None
        users_collection = None
        reports_collection = None
except Exception as e:
    logger.error(f"MongoDB connection failed: {e}")
    client = None
    feedback_collection = None
    medical_input_collection = None
    users_collection = None
    reports_collection = None

# Data classes for better organization
@dataclass
class PredictionResult:
    disease: str
    prediction: int
    probability: float
    risk_level: str
    confidence: float
    timestamp: datetime
    user_input: List[float]

@dataclass
class User:
    username: str
    email: str
    password_hash: str
    created_at: datetime
    last_login: datetime

# Page config with custom styling
st.set_page_config(
    page_title="Advanced Predictive Healthcare Analytics", 
    layout="wide", 
    page_icon="🏥",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
def load_custom_css():
    st.markdown("""
    <style>
    /* Main container styling */
    .main {
        padding: 2rem 1rem;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Card styling */
    .prediction-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    /* Risk level styling */
    .risk-low {
        background: linear-gradient(90deg, #4CAF50, #45a049);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        font-weight: bold;
    }
    
    .risk-moderate {
        background: linear-gradient(90deg, #FF9800, #f57c00);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        font-weight: bold;
    }
    
    .risk-high {
        background: linear-gradient(90deg, #f44336, #d32f2f);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        font-weight: bold;
    }
    
    /* Input form styling */
    .stNumberInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e1e5e9;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Metric styling */
    [data-testid="metric-container"] {
        background: linear-gradient(90deg, #f8f9fa, #e9ecef);
        border: 1px solid #dee2e6;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Load models with error handling
@st.cache_resource
def load_models():
    """Load ML models with caching for better performance"""
    models = {}
    MODEL_DIR = "model"
    
    try:
        if os.path.exists(os.path.join(MODEL_DIR, 'diabetes_model.sav')):
            models['diabetes'] = pickle.load(open(os.path.join(MODEL_DIR, 'diabetes_model.sav'), 'rb'))
        if os.path.exists(os.path.join(MODEL_DIR, 'Heart_model.sav')):
            models['heart_disease'] = pickle.load(open(os.path.join(MODEL_DIR, 'Heart_model.sav'), 'rb'))
        if os.path.exists(os.path.join(MODEL_DIR, 'parkinsons_model.sav')):
            models['parkinsons'] = pickle.load(open(os.path.join(MODEL_DIR, 'parkinsons_model.sav'), 'rb'))
        
        logger.info(f"Loaded {len(models)} models successfully")
        return models
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        st.error("Error loading ML models. Please check model files.")
        return {}

# Authentication functions
class AuthManager:
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        """Verify password against hash"""
        return AuthManager.hash_password(password) == hashed
    
    @staticmethod
    def create_user(username: str, email: str, password: str) -> bool:
        """Create new user account"""
        if not users_collection:
            return False
        
        # Check if user already exists
        if users_collection.find_one({"$or": [{"username": username}, {"email": email}]}):
            return False
        
        user_data = {
            "username": username,
            "email": email,
            "password_hash": AuthManager.hash_password(password),
            "created_at": datetime.now(),
            "last_login": datetime.now()
        }
        
        try:
            users_collection.insert_one(user_data)
            return True
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            return False
    
    @staticmethod
    def authenticate_user(username: str, password: str) -> bool:
        """Authenticate user login"""
        if not users_collection:
            return True  # Skip auth if no DB
        
        user = users_collection.find_one({"username": username})
        if user and AuthManager.verify_password(password, user['password_hash']):
            # Update last login
            users_collection.update_one(
                {"username": username},
                {"$set": {"last_login": datetime.now()}}
            )
            return True
        return False

# Enhanced validation functions
class ValidationManager:
    @staticmethod
    def is_valid_email(email: str) -> bool:
        """Validate email format"""
        email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(email_regex, email) is not None
    
    @staticmethod
    def validate_numeric_input(value: str, min_val: float = None, max_val: float = None) -> Tuple[bool, str]:
        """Validate numeric input with optional range checking"""
        try:
            num_val = float(value)
            if min_val is not None and num_val < min_val:
                return False, f"Value must be at least {min_val}"
            if max_val is not None and num_val > max_val:
                return False, f"Value must be at most {max_val}"
            return True, ""
        except ValueError:
            return False, "Please enter a valid number"
    
    @staticmethod
    def validate_required_fields(fields: Dict[str, str]) -> Tuple[bool, List[str]]:
        """Validate required fields are not empty"""
        errors = []
        for field_name, value in fields.items():
            if not value or str(value).strip() == "":
                errors.append(f"{field_name} is required")
        return len(errors) == 0, errors

# Enhanced visualization functions
class VisualizationManager:
    @staticmethod
    def create_input_radar_chart(user_input: List[float], feature_names: List[str], title: str):
        """Create radar chart for input visualization"""
        if len(user_input) > 8:  # Too many features for radar chart
            return VisualizationManager.create_input_bar_chart(user_input, feature_names, title)
        
        # Normalize values for better visualization
        normalized_input = []
        for val in user_input:
            if val == 0:
                normalized_input.append(0)
            else:
                normalized_input.append(min(val / max(user_input) * 100, 100))
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=normalized_input,
            theta=feature_names,
            fill='toself',
            name='Your Values',
            fillcolor='rgba(102, 126, 234, 0.3)',
            line=dict(color='rgb(102, 126, 234)')
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title=title,
            title_x=0.5,
            height=500
        )
        
        return fig
    
    @staticmethod
    def create_input_bar_chart(user_input: List[float], feature_names: List[str], title: str):
        """Create enhanced bar chart for input visualization"""
        df = pd.DataFrame({
            'Feature': feature_names,
            'Value': user_input
        })
        
        fig = px.bar(
            df, 
            x='Feature', 
            y='Value',
            title=title,
            color='Value',
            color_continuous_scale='viridis',
            text='Value'
        )
        
        fig.update_layout(
            xaxis_tickangle=-45,
            height=500,
            showlegend=False
        )
        
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        
        return fig
    
    @staticmethod
    def create_risk_gauge(risk_score: float, risk_level: str):
        """Create gauge chart for risk visualization"""
        colors = {
            'Low Risk': '#4CAF50',
            'Moderate Risk': '#FF9800', 
            'High Risk': '#f44336'
        }
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = risk_score * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"Risk Assessment: {risk_level}"},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': colors.get(risk_level, '#FF9800')},
                'steps': [
                    {'range': [0, 33], 'color': "lightgray"},
                    {'range': [33, 66], 'color': "gray"},
                    {'range': [66, 100], 'color': "darkgray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(height=400)
        return fig
    
    @staticmethod
    def create_prediction_history_chart(history_data: List[Dict]):
        """Create chart showing prediction history"""
        if not history_data:
            return None
        
        df = pd.DataFrame(history_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        fig = px.line(
            df, 
            x='timestamp', 
            y='probability',
            color='disease',
            title='Prediction History Over Time',
            markers=True
        )
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Risk Probability",
            height=400
        )
        
        return fig

# Enhanced prediction functions
class PredictionManager:
    def __init__(self, models: Dict):
        self.models = models
    
    def predict_with_confidence(self, model_name: str, user_input: List[float]) -> PredictionResult:
        """Make prediction with confidence score"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        try:
            # Make prediction
            prediction = model.predict([user_input])
            
            # Get prediction probabilities if available
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba([user_input])
                probability = max(probabilities[0])
                confidence = probability
            else:
                probability = 0.8 if prediction[0] == 1 else 0.2
                confidence = 0.7
            
            # Calculate risk level
            risk_level = self._calculate_risk_level(model_name, prediction[0], user_input)
            
            return PredictionResult(
                disease=model_name,
                prediction=int(prediction[0]),
                probability=probability,
                risk_level=risk_level,
                confidence=confidence,
                timestamp=datetime.now(),
                user_input=user_input
            )
        
        except Exception as e:
            logger.error(f"Prediction error for {model_name}: {e}")
            raise
    
    def _calculate_risk_level(self, disease_name: str, prediction: int, user_input: List[float]) -> str:
        """Calculate detailed risk level based on disease and input"""
        if disease_name == 'diabetes':
            # Enhanced diabetes risk calculation
            glucose = user_input[1] if len(user_input) > 1 else 0
            bmi = user_input[5] if len(user_input) > 5 else 0
            age = user_input[7] if len(user_input) > 7 else 0
            
            risk_factors = 0
            if glucose > 140: risk_factors += 2
            elif glucose > 100: risk_factors += 1
            
            if bmi > 30: risk_factors += 2
            elif bmi > 25: risk_factors += 1
            
            if age > 45: risk_factors += 1
            
            if prediction == 1:
                if risk_factors >= 4: return "High Risk"
                elif risk_factors >= 2: return "Moderate Risk"
                else: return "Moderate Risk"
            else:
                if risk_factors >= 3: return "Moderate Risk"
                else: return "Low Risk"
        
        elif disease_name == 'heart_disease':
            return "High Risk" if prediction == 1 else "Low Risk"
        
        elif disease_name == 'parkinsons':
            return "High Risk" if prediction == 1 else "Low Risk"
        
        return "Unknown Risk"

# Data management functions
class DataManager:
    @staticmethod
    def save_prediction_result(result: PredictionResult, username: str = "anonymous"):
        """Save prediction result to database"""
        if not medical_input_collection:
            return
        
        try:
            data = {
                "username": username,
                "disease_name": result.disease,
                "input_data": result.user_input,
                "prediction_result": result.prediction,
                "probability": result.probability,
                "risk_level": result.risk_level,
                "confidence": result.confidence,
                "timestamp": result.timestamp
            }
            medical_input_collection.insert_one(data)
            logger.info(f"Saved prediction result for {username}")
        except Exception as e:
            logger.error(f"Error saving prediction: {e}")
    
    @staticmethod
    def get_user_history(username: str, limit: int = 10) -> List[Dict]:
        """Get user's prediction history"""
        if not medical_input_collection:
            return []
        
        try:
            results = list(medical_input_collection.find(
                {"username": username}
            ).sort("timestamp", -1).limit(limit))
            
            for result in results:
                result['_id'] = str(result['_id'])  # Convert ObjectId to string
            
            return results
        except Exception as e:
            logger.error(f"Error retrieving history: {e}")
            return []
    
    @staticmethod
    def export_user_data(username: str) -> pd.DataFrame:
        """Export user data as DataFrame for download"""
        history = DataManager.get_user_history(username, limit=100)
        if not history:
            return pd.DataFrame()
        
        df = pd.DataFrame(history)
        if '_id' in df.columns:
            df = df.drop('_id', axis=1)
        
        return df
    
    @staticmethod
    def save_feedback(name: str, email: str, message: str):
        """Save user feedback"""
        if not feedback_collection:
            return
        
        try:
            feedback_data = {
                "name": name,
                "email": email,
                "message": message,
                "timestamp": datetime.now()
            }
            feedback_collection.insert_one(feedback_data)
        except Exception as e:
            logger.error(f"Error saving feedback: {e}")

# Report generation
class ReportManager:
    @staticmethod
    def generate_comprehensive_report(username: str, prediction_result: PredictionResult) -> Dict:
        """Generate comprehensive health report"""
        history = DataManager.get_user_history(username)
        
        report = {
            "user": username,
            "report_date": datetime.now(),
            "current_prediction": {
                "disease": prediction_result.disease,
                "result": "Positive" if prediction_result.prediction == 1 else "Negative",
                "risk_level": prediction_result.risk_level,
                "confidence": f"{prediction_result.confidence * 100:.1f}%"
            },
            "history_summary": {
                "total_predictions": len(history),
                "diseases_tested": len(set([h['disease_name'] for h in history])),
                "last_test_date": history[0]['timestamp'] if history else None
            },
            "recommendations": ReportManager._generate_recommendations(prediction_result)
        }
        
        return report
    
    @staticmethod
    def _generate_recommendations(result: PredictionResult) -> List[str]:
        """Generate personalized recommendations based on prediction"""
        recommendations = []
        
        if result.disease == 'diabetes':
            if result.prediction == 1:
                recommendations.extend([
                    "Consult with an endocrinologist for diabetes management",
                    "Monitor blood glucose levels regularly",
                    "Follow a diabetes-friendly diet low in refined carbs",
                    "Engage in regular physical activity (150 minutes/week)",
                    "Consider diabetes education classes"
                ])
            else:
                recommendations.extend([
                    "Maintain a healthy weight through balanced diet",
                    "Limit sugary foods and drinks",
                    "Stay physically active with regular exercise",
                    "Get annual health screenings"
                ])
        
        elif result.disease == 'heart_disease':
            if result.prediction == 1:
                recommendations.extend([
                    "Seek immediate consultation with a cardiologist",
                    "Follow a heart-healthy diet (Mediterranean or DASH)",
                    "Take prescribed medications as directed",
                    "Monitor blood pressure regularly",
                    "Quit smoking if applicable"
                ])
            else:
                recommendations.extend([
                    "Maintain healthy cholesterol levels",
                    "Exercise regularly for cardiovascular health",
                    "Limit sodium intake",
                    "Manage stress through relaxation techniques"
                ])
        
        elif result.disease == 'parkinsons':
            if result.prediction == 1:
                recommendations.extend([
                    "Consult with a neurologist specializing in movement disorders",
                    "Consider physical therapy and occupational therapy",
                    "Maintain regular exercise routine",
                    "Join Parkinson's support groups",
                    "Consider speech therapy if needed"
                ])
            else:
                recommendations.extend([
                    "Maintain regular physical activity",
                    "Protect head from injury",
                    "Stay socially and mentally active",
                    "Follow up with regular neurological screenings"
                ])
        
        # General recommendations
        recommendations.extend([
            "Maintain regular follow-ups with healthcare providers",
            "Keep a health diary to track symptoms",
            "Stay informed about your health conditions"
        ])
        
        return recommendations

# Initialize managers
models = load_models()
prediction_manager = PredictionManager(models) if models else None

# Authentication sidebar
def render_auth_sidebar():
    """Render authentication sidebar"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.username = ""
    
    if not st.session_state.authenticated:
        st.sidebar.title("🔐 Authentication")
        
        auth_tab = st.sidebar.radio("Choose action", ["Login", "Sign Up"])
        
        if auth_tab == "Login":
            with st.sidebar.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                login_button = st.form_submit_button("Login")
                
                if login_button:
                    if AuthManager.authenticate_user(username, password):
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.sidebar.success("Logged in successfully!")
                        st.rerun()
                    else:
                        st.sidebar.error("Invalid credentials")
        
        else:  # Sign Up
            with st.sidebar.form("signup_form"):
                new_username = st.text_input("Choose Username")
                email = st.text_input("Email")
                new_password = st.text_input("Choose Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                signup_button = st.form_submit_button("Sign Up")
                
                if signup_button:
                    # Validation
                    if not ValidationManager.is_valid_email(email):
                        st.sidebar.error("Invalid email format")
                    elif new_password != confirm_password:
                        st.sidebar.error("Passwords don't match")
                    elif len(new_password) < 6:
                        st.sidebar.error("Password must be at least 6 characters")
                    elif AuthManager.create_user(new_username, email, new_password):
                        st.sidebar.success("Account created! Please login.")
                    else:
                        st.sidebar.error("Username or email already exists")
    
    else:
        st.sidebar.title(f"👋 Welcome, {st.session_state.username}!")
        if st.sidebar.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.username = ""
            st.rerun()

# Main app
def main():
    load_custom_css()
    render_auth_sidebar()
    
    # Skip auth for demo purposes if no DB
    if not users_collection:
        st.session_state.authenticated = True
        st.session_state.username = "demo_user"
    
    if not st.session_state.authenticated:
        st.markdown("""
        <div class="main-header">
            <h1>🏥 Advanced Predictive Healthcare Analytics</h1>
            <p>Please login or create an account to access the healthcare prediction tools</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🏥 Advanced Predictive Healthcare Analytics</h1>
        <p>AI-powered disease prediction and risk assessment platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation menu
    with st.sidebar:
        selected = option_menu(
            'Healthcare Analytics Suite',
            [
                'Dashboard', 
                'Diabetes Analysis', 
                'Heart Disease Analysis', 
                'Parkinsons Analysis',
                'Comparison Tools',
                'Reports & Export',
                'Health Recommendations', 
                'Contact & Feedback'
            ],
            menu_icon='hospital-fill',
            icons=[
                'speedometer2', 'activity', 'heart', 'person', 
                'bar-chart', 'file-text', 'umbrella', 'envelope'
            ],
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "transparent"},
                "icon": {"color": "white", "font-size": "18px"}, 
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "color": "white"},
                "nav-link-selected": {"background-color": "rgba(255,255,255,0.2)"},
            }
        )
    
    # Route to different pages
    if selected == 'Dashboard':
        render_dashboard()
    elif selected == 'Diabetes Analysis':
        render_diabetes_analysis()
    elif selected == 'Heart Disease Analysis':
        render_heart_disease_analysis()
    elif selected == 'Parkinsons Analysis':
        render_parkinsons_analysis()
    elif selected == 'Comparison Tools':
        render_comparison_tools()
    elif selected == 'Reports & Export':
        render_reports_export()
    elif selected == 'Health Recommendations':
        render_recommendations()
    elif selected == 'Contact & Feedback':
        render_contact_form()

def render_dashboard():
    """Render main dashboard"""
    st.title("📊 Healthcare Analytics Dashboard")
    
    # Get user history for dashboard metrics
    history = DataManager.get_user_history(st.session_state.username)
    
    # Dashboard metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Tests",
            len(history),
            delta=f"+{len([h for h in history if h['timestamp'].date() == datetime.now().date()])}" if history else "0"
        )
    
    with col2:
        diseases_tested = len(set([h['disease_name'] for h in history])) if history else 0
        st.metric("Diseases Tested", diseases_tested)
    
    with col3:
        last_test = history[0]['timestamp'].strftime("%Y-%m-%d") if history else "Never"
        st.metric("Last Test", last_test)
    
    with col4:
        high_risk_count = len([h for h in history if h.get('risk_level') == 'High Risk'])
        st.metric("High Risk Results", high_risk_count)
    
    if history:
        st.markdown("### 📈 Prediction History")
        
        # Create history visualization
        fig = VisualizationManager.create_prediction_history_chart(history)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Recent results table
        st.markdown("### 📋 Recent Test Results")
        recent_history = history[:5]  # Last 5 results
        
        df_display = pd.DataFrame([
            {
                "Date": h['timestamp'].strftime("%Y-%m-%d %H:%M"),
                "Disease": h['disease_name'].replace('_', ' ').title(),
                "Result": "Positive" if h['prediction_result'] == 1 else "Negative",
                "Risk Level": h.get('risk_level', 'Unknown'),
                "Confidence": f"{h.get('confidence', 0.5) * 100:.1f}%"
            }
            for h in recent_history
        ])
        
        st.dataframe(df_display, use_container_width=True)
        
    else:
        st.info("📝 No test history available. Take your first health assessment!")
        
        # Quick start buttons
        st.markdown("### 🚀 Quick Start")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🩺 Diabetes Test", use_container_width=True):
                st.session_state.page = 'Diabetes Analysis'
                st.rerun()
        
        with col2:
            if st.button("❤️ Heart Test", use_container_width=True):
                st.session_state.page = 'Heart Disease Analysis'
                st.rerun()
        
        with col3:
            if st.button("🧠 Parkinson's Test", use_container_width=True):
                st.session_state.page = 'Parkinsons Analysis'
                st.rerun()

# New function to render Diabetes Analysis page
def render_diabetes_analysis():
    st.title("🩺 Diabetes Prediction Analysis")
    st.markdown("### Enter the patient's medical details to predict the risk of Diabetes.")

    # Input form
    col1, col2, col3 = st.columns(3)
    with col1:
        pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=17, value=0)
        glucose = st.number_input('Glucose Level', min_value=0, max_value=200, value=120)
        blood_pressure = st.number_input('Blood Pressure value', min_value=0, max_value=122, value=70)
    with col2:
        skin_thickness = st.number_input('Skin Thickness value', min_value=0, max_value=99, value=20)
        insulin = st.number_input('Insulin Level', min_value=0, max_value=846, value=79)
        bmi = st.number_input('BMI value', min_value=0.0, max_value=67.1, value=32.0, format="%.1f")
    with col3:
        diabetes_pedigree = st.number_input('Diabetes Pedigree Function value', min_value=0.0, max_value=2.42, value=0.47, format="%.3f")
        age = st.number_input('Age of the Person', min_value=0, max_value=120, value=30)
    
    user_input = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]
    feature_names = ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 'Insulin', 'BMI', 'Diabetes Pedigree Function', 'Age']
    
    predict_button = st.button("Predict Diabetes Risk")
    
    if predict_button and prediction_manager:
        with st.spinner("Analyzing data..."):
            time.sleep(2)  # Simulate processing time
            try:
                # Prediction
                result = prediction_manager.predict_with_confidence('diabetes', user_input)
                
                # Save and display
                DataManager.save_prediction_result(result, st.session_state.username)
                display_prediction_results(result, user_input, feature_names)
            except ValueError as e:
                st.error(f"Prediction error: {e}")
            except Exception:
                st.error("An unexpected error occurred during prediction.")
    elif predict_button and not prediction_manager:
        st.warning("Prediction models are not loaded. Please contact support.")

# New function to render Heart Disease Analysis page
def render_heart_disease_analysis():
    st.title("❤️ Heart Disease Prediction Analysis")
    st.markdown("### Enter the patient's medical details to predict the risk of Heart Disease.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input('Age', min_value=1, max_value=120, value=50)
        sex = st.selectbox('Sex', options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
        cp = st.selectbox('Chest Pain Type', options=[0, 1, 2, 3], format_func=lambda x: f"Type {x}")
        trestbps = st.number_input('Resting Blood Pressure', min_value=0, max_value=200, value=120)
        chol = st.number_input('Serum Cholestoral in mg/dl', min_value=0, max_value=600, value=200)
    with col2:
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', options=[1, 0], format_func=lambda x: "True" if x == 1 else "False")
        restecg = st.selectbox('Resting Electrocardiographic results', options=[0, 1, 2])
        thalach = st.number_input('Maximum Heart Rate Achieved', min_value=0, max_value=220, value=150)
        exang = st.selectbox('Exercise Induced Angina', options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    with col3:
        oldpeak = st.number_input('ST depression induced by exercise', min_value=0.0, max_value=6.2, value=1.0, format="%.1f")
        slope = st.selectbox('Slope of the peak exercise ST segment', options=[0, 1, 2])
        ca = st.selectbox('Number of major vessels colored by fluoroscopy', options=[0, 1, 2, 3, 4])
        thal = st.selectbox('Thal', options=[0, 1, 2, 3], format_func=lambda x: f"Thal {x}")

    user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
    feature_names = ['Age', 'Sex', 'Chest Pain Type', 'Resting BP', 'Cholesterol', 'Fasting BS', 'ECG Results', 'Max Heart Rate', 'Exercise Angina', 'ST Depression', 'ST Slope', 'CA', 'Thal']

    predict_button = st.button("Predict Heart Disease Risk")
    
    if predict_button and prediction_manager:
        with st.spinner("Analyzing data..."):
            time.sleep(2)
            try:
                result = prediction_manager.predict_with_confidence('heart_disease', user_input)
                DataManager.save_prediction_result(result, st.session_state.username)
                display_prediction_results(result, user_input, feature_names)
            except ValueError as e:
                st.error(f"Prediction error: {e}")
            except Exception:
                st.error("An unexpected error occurred during prediction.")
    elif predict_button and not prediction_manager:
        st.warning("Prediction models are not loaded. Please contact support.")

# New function to render Parkinsons Analysis page
def render_parkinsons_analysis():
    st.title("🧠 Parkinson's Disease Prediction Analysis")
    st.markdown("### Enter the patient's voice details to predict the risk of Parkinson's Disease.")

    col1, col2 = st.columns(2)
    with col1:
        fo = st.number_input('MDVP:Fo(Hz)', min_value=0.0, value=119.992)
        fhi = st.number_input('MDVP:Fhi(Hz)', min_value=0.0, value=157.302)
        flo = st.number_input('MDVP:Flo(Hz)', min_value=0.0, value=74.997)
        jitter_percent = st.number_input('MDVP:Jitter(%)', min_value=0.0, value=0.00784)
        jitter_abs = st.number_input('MDVP:Jitter(Abs)', min_value=0.0, value=0.00007)
        rap = st.number_input('MDVP:RAP', min_value=0.0, value=0.00370)
        ppq = st.number_input('MDVP:PPQ', min_value=0.0, value=0.00554)
        ddp = st.number_input('Jitter:DDP', min_value=0.0, value=0.01109)
        shimmer = st.number_input('MDVP:Shimmer', min_value=0.0, value=0.04374)
        shimmer_db = st.number_input('MDVP:Shimmer(dB)', min_value=0.0, value=0.426)
        apq3 = st.number_input('Shimmer:APQ3', min_value=0.0, value=0.02182)
    with col2:
        apq5 = st.number_input('Shimmer:APQ5', min_value=0.0, value=0.03130)
        apq = st.number_input('MDVP:APQ', min_value=0.0, value=0.02971)
        dda = st.number_input('Shimmer:DDA', min_value=0.0, value=0.06545)
        nhr = st.number_input('NHR', min_value=0.0, value=0.02211)
        hnr = st.number_input('HNR', min_value=0.0, value=21.033)
        rpde = st.number_input('RPDE', min_value=0.0, max_value=1.0, value=0.414783)
        dfa = st.number_input('DFA', min_value=0.0, max_value=1.0, value=0.815285)
        spread1 = st.number_input('spread1', min_value=-10.0, value=-4.813031)
        spread2 = st.number_input('spread2', min_value=0.0, max_value=1.0, value=0.266482)
        d2 = st.number_input('D2', min_value=0.0, max_value=10.0, value=2.301442)
        ppe = st.number_input('PPE', min_value=0.0, max_value=1.0, value=0.284654)

    user_input = [fo, fhi, flo, jitter_percent, jitter_abs, rap, ppq, ddp, shimmer, shimmer_db, apq3, apq5, apq, dda, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe]
    feature_names = ['Fo(Hz)', 'Fhi(Hz)', 'Flo(Hz)', 'Jitter(%)', 'Jitter(Abs)', 'RAP', 'PPQ', 'DDP', 'Shimmer', 'Shimmer(dB)', 'APQ3', 'APQ5', 'APQ', 'DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE']
    
    predict_button = st.button("Predict Parkinson's Risk")
    
    if predict_button and prediction_manager:
        with st.spinner("Analyzing data..."):
            time.sleep(2)
            try:
                result = prediction_manager.predict_with_confidence('parkinsons', user_input)
                DataManager.save_prediction_result(result, st.session_state.username)
                display_prediction_results(result, user_input, feature_names)
            except ValueError as e:
                st.error(f"Prediction error: {e}")
            except Exception:
                st.error("An unexpected error occurred during prediction.")
    elif predict_button and not prediction_manager:
        st.warning("Prediction models are not loaded. Please contact support.")

# New function to render Comparison Tools page
def render_comparison_tools():
    st.title("🔬 Prediction Comparison Tools")
    st.markdown("### Compare two different prediction scenarios or your current result with a historical one.")
    
    history = DataManager.get_user_history(st.session_state.username)
    if not history:
        st.warning("No prediction history available for comparison.")
        return
    
    history_df = pd.DataFrame(history)
    history_df['display_label'] = history_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M') + ' - ' + history_df['disease_name'].str.title().str.replace('_', ' ')
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Scenario 1")
        scenario1_option = st.selectbox("Select Prediction for Comparison 1", options=history_df['display_label'], key="scenario1")
        if scenario1_option:
            scenario1_data = history_df[history_df['display_label'] == scenario1_option].iloc[0]
            st.info(f"Disease: {scenario1_data['disease_name'].title().replace('_', ' ')}\nRisk Level: {scenario1_data['risk_level']}\nProbability: {scenario1_data['probability']:.2f}")

    with col2:
        st.subheader("Scenario 2")
        scenario2_option = st.selectbox("Select Prediction for Comparison 2", options=history_df['display_label'], key="scenario2")
        if scenario2_option:
            scenario2_data = history_df[history_df['display_label'] == scenario2_option].iloc[0]
            st.info(f"Disease: {scenario2_data['disease_name'].title().replace('_', ' ')}\nRisk Level: {scenario2_data['risk_level']}\nProbability: {scenario2_data['probability']:.2f}")

    if st.button("Compare Scenarios"):
        if scenario1_option and scenario2_option:
            data_s1 = history_df[history_df['display_label'] == scenario1_option].iloc[0]
            data_s2 = history_df[history_df['display_label'] == scenario2_option].iloc[0]
            
            st.markdown("### Comparison Results")
            comparison_df = pd.DataFrame({
                "Metric": ["Disease", "Risk Level", "Probability", "Input Data"],
                "Scenario 1": [data_s1['disease_name'].title().replace('_', ' '), data_s1['risk_level'], f"{data_s1['probability']:.2f}", data_s1['input_data']],
                "Scenario 2": [data_s2['disease_name'].title().replace('_', ' '), data_s2['risk_level'], f"{data_s2['probability']:.2f}", data_s2['input_data']]
            })
            
            st.dataframe(comparison_df)

            # Additional visualization if same disease is compared
            if data_s1['disease_name'] == data_s2['disease_name']:
                st.subheader(f"Input Feature Comparison for {data_s1['disease_name'].title().replace('_', ' ')}")
                
                # Get feature names for the specific disease
                if data_s1['disease_name'] == 'diabetes':
                    feature_names = ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 'Insulin', 'BMI', 'Diabetes Pedigree Function', 'Age']
                elif data_s1['disease_name'] == 'heart_disease':
                    feature_names = ['Age', 'Sex', 'Chest Pain Type', 'Resting BP', 'Cholesterol', 'Fasting BS', 'ECG Results', 'Max Heart Rate', 'Exercise Angina', 'ST Depression', 'ST Slope', 'CA', 'Thal']
                elif data_s1['disease_name'] == 'parkinsons':
                    feature_names = ['Fo(Hz)', 'Fhi(Hz)', 'Flo(Hz)', 'Jitter(%)', 'Jitter(Abs)', 'RAP', 'PPQ', 'DDP', 'Shimmer', 'Shimmer(dB)', 'APQ3', 'APQ5', 'APQ', 'DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE']
                else:
                    feature_names = [f"Feature {i+1}" for i in range(len(data_s1['input_data']))]
                
                compare_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Scenario 1': data_s1['input_data'],
                    'Scenario 2': data_s2['input_data']
                })

                fig = px.bar(compare_df, x='Feature', y=['Scenario 1', 'Scenario 2'],
                             barmode='group', title="Input Feature Comparison")
                st.plotly_chart(fig, use_container_width=True)


# New function to render Reports & Export page
def render_reports_export():
    st.title("📄 Reports & Data Export")
    st.markdown("### Generate and download detailed health reports and your historical data.")
    
    history_data = DataManager.get_user_history(st.session_state.username)
    if not history_data:
        st.warning("No historical data available to generate reports or export.")
        return

    st.markdown("---")
    
    st.subheader("Generate Comprehensive Health Report")
    
    # Select a specific prediction to generate a report
    history_df = pd.DataFrame(history_data)
    history_df['display_label'] = history_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M') + ' - ' + history_df['disease_name'].str.title().str.replace('_', ' ')
    
    selected_report = st.selectbox("Select a prediction to generate a detailed report:", options=history_df['display_label'])
    
    if selected_report:
        selected_data = history_df[history_df['display_label'] == selected_report].iloc[0]
        
        # Create a dummy PredictionResult object
        prediction_result = PredictionResult(
            disease=selected_data['disease_name'],
            prediction=selected_data['prediction_result'],
            probability=selected_data['probability'],
            risk_level=selected_data['risk_level'],
            confidence=selected_data['confidence'],
            timestamp=selected_data['timestamp'],
            user_input=selected_data['input_data']
        )
        
        if st.button("Generate Report"):
            with st.spinner("Generating report..."):
                report = ReportManager.generate_comprehensive_report(st.session_state.username, prediction_result)
                
                st.markdown("### Generated Report Preview")
                
                # Display report in a structured way
                st.json(report)
                
                # Save as JSON for download
                report_json = json.dumps(report, indent=4, default=str)
                st.download_button(
                    label="Download Report as JSON",
                    data=report_json,
                    file_name=f"health_report_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json"
                )

    st.markdown("---")
    
    st.subheader("Export Raw Data")
    st.markdown("Download your entire prediction history as a CSV file.")
    
    export_df = DataManager.export_user_data(st.session_state.username)
    if not export_df.empty:
        csv = export_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Data as CSV",
            data=csv,
            file_name=f"healthcare_data_export_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No data available to export.")
        
# New function for displaying detailed results
def display_prediction_results(result: PredictionResult, user_input: List[float], feature_names: List[str]):
    st.success("Analysis Complete!")
    
    # Display result in a card
    st.markdown(f"""
    <div class="prediction-card">
        <h3>Prediction Result for {result.disease.title().replace('_', ' ')}</h3>
        <p>Your prediction result is: <b>{'Positive (You may have the disease)' if result.prediction == 1 else 'Negative (You are unlikely to have the disease)'}</b></p>
        <p>Based on our analysis, your risk level is: <span class="risk-{result.risk_level.lower().replace(' ', '-')}">{result.risk_level}</span></p>
        <p>Model Confidence: <b>{result.confidence * 100:.2f}%</b></p>
        <p><i>Note: This is an AI-generated prediction. Always consult a medical professional for diagnosis.</i></p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("Detailed Analysis & Visualizations")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Your Input Values")
        # Display input data table
        input_df = pd.DataFrame({
            "Feature": feature_names,
            "Value": user_input
        })
        st.dataframe(input_df, use_container_width=True)
    
    with col2:
        st.markdown("#### Risk Assessment Gauge")
        # Display risk gauge chart
        gauge_fig = VisualizationManager.create_risk_gauge(result.probability, result.risk_level)
        st.plotly_chart(gauge_fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Input Feature Visualization")
    # Display radar or bar chart
    if len(user_input) <= 8:
        radar_fig = VisualizationManager.create_input_radar_chart(user_input, feature_names, "Your Input Values (Normalized)")
        st.plotly_chart(radar_fig, use_container_width=True)
    else:
        bar_fig = VisualizationManager.create_input_bar_chart(user_input, feature_names, "Your Input Values")
        st.plotly_chart(bar_fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Personalized Health Recommendations")
    recommendations = ReportManager._generate_recommendations(result)
    for rec in recommendations:
        st.markdown(f"✅ {rec}")

# New function to render Health Recommendations page
def render_recommendations():
    st.title("💡 Health & Wellness Recommendations")
    st.markdown("### Get general health advice and personalized recommendations based on your recent test results.")
    
    history_data = DataManager.get_user_history(st.session_state.username, limit=1)
    
    if history_data:
        latest_result = PredictionResult(
            disease=history_data[0]['disease_name'],
            prediction=history_data[0]['prediction_result'],
            probability=history_data[0]['probability'],
            risk_level=history_data[0]['risk_level'],
            confidence=history_data[0]['confidence'],
            timestamp=history_data[0]['timestamp'],
            user_input=history_data[0]['input_data']
        )
        
        st.subheader(f"Recommendations based on your last test for {latest_result.disease.title().replace('_', ' ')}")
        recommendations = ReportManager._generate_recommendations(latest_result)
        for rec in recommendations:
            st.markdown(f"✅ {rec}")
    else:
        st.info("No recent test results found. Please perform a test to get personalized recommendations.")

    st.markdown("---")
    st.subheader("General Wellness Tips")
    st.markdown("""
    - **Balanced Diet:** Eat a variety of fruits, vegetables, whole grains, lean proteins, and healthy fats.
    - **Regular Exercise:** Aim for at least 150 minutes of moderate-intensity aerobic exercise or 75 minutes of vigorous-intensity exercise per week.
    - **Quality Sleep:** Get 7-9 hours of sleep per night to support physical and mental health.
    - **Stress Management:** Practice mindfulness, meditation, or yoga to reduce stress levels.
    - **Hydration:** Drink plenty of water throughout the day.
    - **Regular Check-ups:** Schedule routine medical check-ups to monitor your health.
    """)


# New function to render Contact & Feedback page
def render_contact_form():
    st.title("📧 Contact & Feedback")
    st.markdown("### We value your feedback! Please let us know how we can improve.")

    with st.form("feedback_form"):
        name = st.text_input("Your Name")
        email = st.text_input("Your Email")
        message = st.text_area("Your Message")
        submit_button = st.form_submit_button("Submit Feedback")

        if submit_button:
            if not ValidationManager.is_valid_email(email):
                st.error("Please enter a valid email address.")
            elif not name or not message:
                st.error("Name and Message fields are required.")
            else:
                try:
                    DataManager.save_feedback(name, email, message)
                    st.success("Thank you for your feedback! It has been submitted successfully.")
                except Exception:
                    st.error("An error occurred while submitting feedback. Please try again later.")

if __name__ == '__main__':
    main()
