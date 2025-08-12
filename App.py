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
        # In a real app, this would be based on min/max of the original training data.
        # For now, we'll do a simple normalization.
        max_val = max(user_input) if any(v > 0 for v in user_input) else 1
        normalized_input = [min(val / max_val, 1) * 100 for val in user_input]
        
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
            color='disease_name',
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
            if prediction == 1:
                return "High Risk"
            else:
                # Add more nuance for heart disease risk
                age = user_input[0] if len(user_input) > 0 else 0
                cholesterol = user_input[9] if len(user_input) > 9 else 0
                if age > 50 and cholesterol > 240:
                    return "Moderate Risk"
                return "Low Risk"
        
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
    # Initialize session state for authentication
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.username = ""
    
    if not st.session_state.authenticated:
        st.sidebar.title("🔐 User Authentication")
        
        # Use tabs for a cleaner UI
        login_tab, signup_tab = st.sidebar.tabs(["Login", "Sign Up"])

        with login_tab:
            st.markdown("### Existing User Login")
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                login_button = st.form_submit_button("Login")
                
                if login_button:
                    if not username or not password:
                        st.error("Username and password are required.")
                    elif AuthManager.authenticate_user(username, password):
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.success("Logged in successfully!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")
        
        with signup_tab:
            st.markdown("### New User Sign Up")
            with st.form("signup_form"):
                new_username = st.text_input("Choose Username")
                email = st.text_input("Email")
                new_password = st.text_input("Choose Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                signup_button = st.form_submit_button("Create Account")
                
                if signup_button:
                    # Validation
                    if not new_username or not email or not new_password or not confirm_password:
                        st.error("All fields are required.")
                    elif not ValidationManager.is_valid_email(email):
                        st.error("Invalid email format.")
                    elif new_password != confirm_password:
                        st.error("Passwords do not match.")
                    elif len(new_password) < 6:
                        st.error("Password must be at least 6 characters.")
                    elif AuthManager.create_user(new_username, email, new_password):
                        st.success("Account created successfully! Please login.")
                        st.rerun()
                    else:
                        st.error("Username or email already exists.")
    
    else:
        # Display user info and logout button if authenticated
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
                # Make prediction
                result = prediction_manager.predict_with_confidence('diabetes', user_input)
                
                # Save result
                DataManager.save_prediction_result(result, st.session_state.username)
                
                # Display results
                st.markdown("---")
                st.markdown(f"## 🩺 Prediction Result: **{result.disease.title()}**")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(VisualizationManager.create_risk_gauge(result.probability, result.risk_level), use_container_width=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h3>Prediction Details</h3>
                        <p>Your model prediction indicates a **{"Positive" if result.prediction == 1 else "Negative"}** result.</p>
                        <p>The confidence of this prediction is: **{result.confidence*100:.2f}%**</p>
                        <p>Your current risk level is: <span class="risk-{result.risk_level.split()[0].lower()}">{result.risk_level}</span></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("### Your Input Data")
                    st.dataframe(pd.DataFrame([user_input], columns=feature_names), use_container_width=True)
                
                st.markdown("---")
                st.markdown("### 📊 Input Feature Analysis")
                fig = VisualizationManager.create_input_radar_chart(user_input, feature_names, "Your Input Values")
                st.plotly_chart(fig, use_container_width=True)

                st.success("Prediction complete! See your full health report in the 'Reports & Export' section.")

            except ValueError as ve:
                st.error(f"Prediction error: {ve}. Please ensure your model files are correct.")
            except Exception as e:
                st.error(f"An unexpected error occurred during prediction: {e}")
    elif predict_button and not prediction_manager:
        st.error("ML models are not loaded. Please check the `model` folder and ensure the files exist.")

def render_heart_disease_analysis():
    st.title("❤️ Heart Disease Prediction Analysis")
    st.markdown("### Enter the patient's medical details to predict the risk of Heart Disease.")
    
    # Input form for Heart Disease
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        age = st.number_input('Age', min_value=1, max_value=120, value=52)
        sex = st.selectbox('Sex', options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
        cp = st.selectbox('Chest Pain Type', options=[0, 1, 2, 3])
        trestbps = st.number_input('Resting Blood Pressure', min_value=80, max_value=200, value=120)
    with col2:
        chol = st.number_input('Serum Cholestoral in mg/dl', min_value=100, max_value=600, value=230)
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        restecg = st.selectbox('Resting Electrocardiographic Results', options=[0, 1, 2])
        thalach = st.number_input('Maximum Heart Rate Achieved', min_value=70, max_value=220, value=150)
    with col3:
        exang = st.selectbox('Exercise Induced Angina', options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        oldpeak = st.number_input('ST depression induced by exercise relative to rest', min_value=0.0, max_value=6.2, value=1.0, format="%.1f")
        slope = st.selectbox('Slope of the peak exercise ST segment', options=[0, 1, 2])
        ca = st.number_input('Number of major vessels (0-3) colored by flourosopy', min_value=0, max_value=4, value=0)
    with col4:
        thal = st.selectbox('Thal', options=[0, 1, 2, 3])
        
    user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
    feature_names = ['Age', 'Sex', 'Chest Pain Type', 'Resting BP', 'Cholesterol', 'Fasting Blood Sugar', 'Resting ECG', 'Max Heart Rate', 'Exercise Angina', 'Oldpeak', 'Slope', 'CA', 'Thal']
    
    predict_button = st.button("Predict Heart Disease Risk")

    if predict_button and prediction_manager:
        with st.spinner("Analyzing data..."):
            time.sleep(2)
            try:
                result = prediction_manager.predict_with_confidence('heart_disease', user_input)
                DataManager.save_prediction_result(result, st.session_state.username)
                
                st.markdown("---")
                st.markdown(f"## ❤️ Prediction Result: **{result.disease.title()}**")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(VisualizationManager.create_risk_gauge(result.probability, result.risk_level), use_container_width=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h3>Prediction Details</h3>
                        <p>Your model prediction indicates a **{"Positive" if result.prediction == 1 else "Negative"}** result.</p>
                        <p>The confidence of this prediction is: **{result.confidence*100:.2f}%**</p>
                        <p>Your current risk level is: <span class="risk-{result.risk_level.split()[0].lower()}">{result.risk_level}</span></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("### Your Input Data")
                    st.dataframe(pd.DataFrame([user_input], columns=feature_names), use_container_width=True)
                
                st.markdown("---")
                st.markdown("### 📊 Input Feature Analysis")
                fig = VisualizationManager.create_input_bar_chart(user_input, feature_names, "Your Input Values")
                st.plotly_chart(fig, use_container_width=True)
                
                st.success("Prediction complete! See your full heart health report in the 'Reports & Export' section.")

            except ValueError as ve:
                st.error(f"Prediction error: {ve}. Please ensure your model files are correct.")
            except Exception as e:
                st.error(f"An unexpected error occurred during prediction: {e}")
    elif predict_button and not prediction_manager:
        st.error("ML models are not loaded. Please check the `model` folder and ensure the files exist.")

def render_parkinsons_analysis():
    st.title("🧠 Parkinson's Disease Prediction Analysis")
    st.markdown("### Enter the patient's voice data parameters to predict the risk of Parkinson's Disease.")
    
    # Input form for Parkinson's
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        fo = st.number_input('MDVP:Fo(Hz)', min_value=88.0, max_value=260.0, value=150.0)
        fhi = st.number_input('MDVP:Fhi(Hz)', min_value=100.0, max_value=600.0, value=200.0)
        flo = st.number_input('MDVP:Flo(Hz)', min_value=65.0, max_value=200.0, value=100.0)
    with col2:
        jitter_abs = st.number_input('Jitter(Abs)', min_value=0.00001, max_value=0.0005, value=0.00005, format="%.5f")
        rap = st.number_input('RAP', min_value=0.0005, max_value=0.004, value=0.001, format="%.4f")
        ppq = st.number_input('PPQ', min_value=0.0005, max_value=0.004, value=0.001, format="%.4f")
    with col3:
        shimmer = st.number_input('Shimmer', min_value=0.01, max_value=0.1, value=0.04, format="%.3f")
        shimmer_db = st.number_input('Shimmer(dB)', min_value=0.1, max_value=1.5, value=0.4, format="%.2f")
        apq3 = st.number_input('APQ3', min_value=0.005, max_value=0.05, value=0.02, format="%.3f")
    with col4:
        apq5 = st.number_input('APQ5', min_value=0.005, max_value=0.05, value=0.02, format="%.3f")
        apq = st.number_input('APQ', min_value=0.01, max_value=0.07, value=0.03, format="%.3f")
        dda = st.number_input('DDA', min_value=0.005, max_value=0.01, value=0.005, format="%.4f")
    with col5:
        nhr = st.number_input('NHR', min_value=0.005, max_value=0.05, value=0.02, format="%.3f")
        hnr = st.number_input('HNR', min_value=10.0, max_value=35.0, value=20.0)
        rpde = st.number_input('RPDE', min_value=0.3, max_value=0.7, value=0.5, format="%.3f")
        dfa = st.number_input('DFA', min_value=0.5, max_value=0.9, value=0.7, format="%.3f")
        spread1 = st.number_input('spread1', min_value=-8.0, max_value=-2.0, value=-5.0, format="%.3f")
        spread2 = st.number_input('spread2', min_value=0.0, max_value=0.5, value=0.2, format="%.3f")
        d2 = st.number_input('D2', min_value=1.0, max_value=4.0, value=2.0, format="%.3f")
        ppe = st.number_input('PPE', min_value=0.0, max_value=0.6, value=0.2, format="%.3f")
    
    user_input = [fo, fhi, flo, 0.00001, jitter_abs, rap, ppq, shimmer, shimmer_db, apq3, apq5, apq, dda, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe] # Reorder to match model
    feature_names = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'Jitter(%)', 'Jitter(Abs)', 'RAP', 'PPQ', 'Shimmer', 'Shimmer(dB)', 'APQ3', 'APQ5', 'APQ', 'DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE']
    
    predict_button = st.button("Predict Parkinson's Risk")
    
    if predict_button and prediction_manager:
        with st.spinner("Analyzing data..."):
            time.sleep(2)
            try:
                result = prediction_manager.predict_with_confidence('parkinsons', user_input)
                DataManager.save_prediction_result(result, st.session_state.username)

                st.markdown("---")
                st.markdown(f"## 🧠 Prediction Result: **{result.disease.title()}**")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(VisualizationManager.create_risk_gauge(result.probability, result.risk_level), use_container_width=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h3>Prediction Details</h3>
                        <p>Your model prediction indicates a **{"Positive" if result.prediction == 1 else "Negative"}** result.</p>
                        <p>The confidence of this prediction is: **{result.confidence*100:.2f}%**</p>
                        <p>Your current risk level is: <span class="risk-{result.risk_level.split()[0].lower()}">{result.risk_level}</span></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("### Your Input Data")
                    st.dataframe(pd.DataFrame([user_input], columns=feature_names), use_container_width=True)
                
                st.markdown("---")
                st.markdown("### 📊 Input Feature Analysis")
                fig = VisualizationManager.create_input_bar_chart(user_input, feature_names, "Your Input Values")
                st.plotly_chart(fig, use_container_width=True)
                
                st.success("Prediction complete! See your full Parkinson's report in the 'Reports & Export' section.")

            except ValueError as ve:
                st.error(f"Prediction error: {ve}. Please ensure your model files are correct.")
            except Exception as e:
                st.error(f"An unexpected error occurred during prediction: {e}")
    elif predict_button and not prediction_manager:
        st.error("ML models are not loaded. Please check the `model` folder and ensure the files exist.")

def render_comparison_tools():
    st.title("🔬 Comparison Tools")
    st.markdown("### Compare two of your previous test results side-by-side.")

    history = DataManager.get_user_history(st.session_state.username, limit=50)
    if len(history) < 2:
        st.info("You need at least two test results to use this tool.")
        return
    
    # Get a list of tests for the dropdowns
    test_options = {f"{h['disease_name'].title()} - {h['timestamp'].strftime('%Y-%m-%d %H:%M')}": h for h in history}
    options_list = list(test_options.keys())
    
    col1, col2 = st.columns(2)
    with col1:
        selected_test1 = st.selectbox("Select Test 1", options=options_list, key="test1_select")
    with col2:
        selected_test2 = st.selectbox("Select Test 2", options=options_list, index=1, key="test2_select")

    if selected_test1 and selected_test2:
        data1 = test_options[selected_test1]
        data2 = test_options[selected_test2]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("---")
            st.subheader(f"Test 1: {data1['disease_name'].title()}")
            st.markdown(f"**Date:** {data1['timestamp'].strftime('%Y-%m-%d %H:%M')}")
            st.markdown(f"**Prediction:** {'Positive' if data1['prediction_result'] == 1 else 'Negative'}")
            st.markdown(f"**Risk Level:** <span class='risk-{data1['risk_level'].split()[0].lower()}'>{data1['risk_level']}</span>", unsafe_allow_html=True)
            
            # Show a chart for Test 1
            if len(data1['input_data']) > 8:
                fig = VisualizationManager.create_input_bar_chart(data1['input_data'], ['Feature']*len(data1['input_data']), "Test 1 Inputs")
            else:
                fig = VisualizationManager.create_input_radar_chart(data1['input_data'], ['Feature']*len(data1['input_data']), "Test 1 Inputs")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("---")
            st.subheader(f"Test 2: {data2['disease_name'].title()}")
            st.markdown(f"**Date:** {data2['timestamp'].strftime('%Y-%m-%d %H:%M')}")
            st.markdown(f"**Prediction:** {'Positive' if data2['prediction_result'] == 1 else 'Negative'}")
            st.markdown(f"**Risk Level:** <span class='risk-{data2['risk_level'].split()[0].lower()}'>{data2['risk_level']}</span>", unsafe_allow_html=True)

            # Show a chart for Test 2
            if len(data2['input_data']) > 8:
                fig = VisualizationManager.create_input_bar_chart(data2['input_data'], ['Feature']*len(data2['input_data']), "Test 2 Inputs")
            else:
                fig = VisualizationManager.create_input_radar_chart(data2['input_data'], ['Feature']*len(data2['input_data']), "Test 2 Inputs")
            st.plotly_chart(fig, use_container_width=True)

def render_reports_export():
    st.title("📄 Reports & Data Export")
    st.markdown("### Generate and download detailed reports of your health predictions.")

    history = DataManager.get_user_history(st.session_state.username)
    if not history:
        st.info("You must complete at least one prediction to generate a report.")
        return

    st.markdown("---")
    st.subheader("📋 Last Prediction Report")
    last_result_dict = history[0]
    
    # Create a PredictionResult object from the stored dictionary
    last_result = PredictionResult(
        disease=last_result_dict['disease_name'],
        prediction=last_result_dict['prediction_result'],
        probability=last_result_dict['probability'],
        risk_level=last_result_dict['risk_level'],
        confidence=last_result_dict['confidence'],
        timestamp=last_result_dict['timestamp'],
        user_input=last_result_dict['input_data']
    )
    
    report = ReportManager.generate_comprehensive_report(st.session_state.username, last_result)
    
    st.json(report)
    
    st.markdown("---")
    st.subheader("📥 Export Your Data")
    
    df = DataManager.export_user_data(st.session_state.username)
    if not df.empty:
        st.markdown("Here is a preview of your data. You can download it as a CSV file.")
        st.dataframe(df, use_container_width=True)
        csv_file = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Data as CSV",
            data=csv_file,
            file_name=f"{st.session_state.username}_health_data.csv",
            mime="text/csv"
        )
    else:
        st.info("No data to export.")

def render_recommendations():
    st.title("💡 Health Recommendations")
    st.markdown("### Personalized health advice based on your last prediction.")

    history = DataManager.get_user_history(st.session_state.username)
    if not history:
        st.info("Complete a prediction to receive personalized recommendations.")
        return
        
    last_result_dict = history[0]
    last_result = PredictionResult(
        disease=last_result_dict['disease_name'],
        prediction=last_result_dict['prediction_result'],
        probability=last_result_dict['probability'],
        risk_level=last_result_dict['risk_level'],
        confidence=last_result_dict['confidence'],
        timestamp=last_result_dict['timestamp'],
        user_input=last_result_dict['input_data']
    )
    
    st.markdown(f"Based on your last **{last_result.disease.title()}** prediction, here are some key recommendations:")
    st.markdown(f"**Result:** {'Positive' if last_result.prediction == 1 else 'Negative'}")
    st.markdown(f"**Risk Level:** <span class='risk-{last_result.risk_level.split()[0].lower()}'>{last_result.risk_level}</span>", unsafe_allow_html=True)
    
    recommendations = ReportManager._generate_recommendations(last_result)
    
    st.markdown("---")
    st.subheader("Action Plan:")
    for rec in recommendations:
        st.markdown(f"- {rec}")
    
    st.info("Disclaimer: These recommendations are for informational purposes only and are not a substitute for professional medical advice. Always consult a qualified healthcare provider for diagnosis and treatment.")

def render_contact_form():
    st.title("✉️ Contact & Feedback")
    st.markdown("### We value your feedback. Please use the form below to get in touch.")
    
    with st.form("feedback_form"):
        name = st.text_input("Your Name")
        email = st.text_input("Your Email")
        message = st.text_area("Your Message", height=200)
        submit_button = st.form_submit_button("Submit Feedback")
        
        if submit_button:
            if not name or not email or not message:
                st.error("Please fill in all fields.")
            elif not ValidationManager.is_valid_email(email):
                st.error("Please enter a valid email address.")
            else:
                DataManager.save_feedback(name, email, message)
                st.success("Thank you for your feedback! We will get back to you soon.")

if __name__ == '__main__':
    # Add this session state check to handle multi-page navigation correctly
    if 'page' not in st.session_state:
        st.session_state.page = 'Dashboard'
        
    main()
