# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

# Configure the page
st.set_page_config(
    page_title="PredSeeker - Developer Employment Predictor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern clean UI with proper contrast
st.markdown("""
<style>
    /* Main app styling */
    .stApp {
        background-color: #ffffff;
        color: #212529;
    }

    /* Top navigation bar - remove dark header */
    .css-18e3th9, .css-1d391kg, header[data-testid="stHeader"] {
        background-color: #ffffff !important;
        border-bottom: 1px solid #dee2e6 !important;
    }

    /* Hide Streamlit menu and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}

    /* App header container */
    .block-container {
        padding-top: 2rem;
    }

    /* Override Streamlit's default text colors */
    .stMarkdown, .stText, p, div, span {
        color: #212529 !important;
    }

    /* Header styling */
    .main-header {
        font-size: 3.5rem;
        color: #2E86AB !important;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }

    .sub-header {
        font-size: 1.5rem;
        color: #495057 !important;
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 400;
    }

    /* Prediction result boxes */
    .prediction-box {
        padding: 2rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.15);
        text-align: center;
    }

    .employed {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 3px solid #28a745;
        color: #155724 !important;
    }

    .unemployed {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border: 3px solid #dc3545;
        color: #721c24 !important;
    }

    .prediction-box h2 {
        margin-bottom: 1rem;
        font-size: 2rem;
        font-weight: 600;
        color: inherit !important;
    }

    .prediction-box h3 {
        margin-bottom: 0.5rem;
        font-size: 1.5rem;
        font-weight: 500;
        color: inherit !important;
    }

    .prediction-box p {
        color: inherit !important;
        font-size: 1.1rem;
    }

    /* Skill category styling */
    .skill-category {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #007bff;
    }

    /* Sidebar styling - Fix dark background issue */
    .css-1d391kg, .css-1cypcdb, .css-17eq0hr, .css-1544g2n {
        background-color: #f8f9fa !important;
    }

    /* Sidebar content styling */
    .css-1d391kg .stMarkdown, .css-1d391kg .stMarkdown p, .css-1d391kg .stMarkdown h1,
    .css-1d391kg .stMarkdown h2, .css-1d391kg .stMarkdown h3, .css-1d391kg .stMarkdown li {
        color: #212529 !important;
    }

    /* Sidebar metric styling */
    .css-1d391kg .metric-container {
        background-color: #ffffff !important;
        border: 1px solid #dee2e6 !important;
        border-radius: 8px !important;
        padding: 0.75rem !important;
        margin: 0.5rem 0 !important;
    }

    /* Sidebar header styling */
    .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3 {
        color: #2E86AB !important;
        border-bottom: 2px solid #dee2e6;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }

    /* Streamlit component overrides - Enhanced */
    .stSelectbox label, .stCheckbox label, .stSlider label, .stNumberInput label, .stRadio label {
        color: #212529 !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        margin-bottom: 0.5rem !important;
        text-shadow: none !important;
    }

    /* Form section labels */
    h2, h3, .css-10trblm {
        color: #2E86AB !important;
        font-weight: 600 !important;
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
        border-bottom: 2px solid #dee2e6 !important;
        padding-bottom: 0.5rem !important;
    }

    .stMetric label {
        color: #495057 !important;
    }

    .stMetric [data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }

    /* Success/Warning/Info boxes */
    .stSuccess {
        background-color: #d1ecf1 !important;
        border: 2px solid #17a2b8 !important;
        border-radius: 8px;
        color: #0c5460 !important;
    }

    .stSuccess .stMarkdown p {
        color: #0c5460 !important;
    }

    .stWarning {
        background-color: #fff3cd !important;
        border: 2px solid #ffc107 !important;
        border-radius: 8px;
        color: #856404 !important;
    }

    .stWarning .stMarkdown p {
        color: #856404 !important;
    }

    .stInfo {
        background-color: #cce7ff !important;
        border: 2px solid #007bff !important;
        border-radius: 8px;
        color: #004085 !important;
    }

    .stInfo .stMarkdown p {
        color: #004085 !important;
    }

    /* Button styling - Enhanced */
    .stButton > button {
        background: linear-gradient(135deg, #007bff 0%, #0056b3 100%) !important;
        color: white !important;
        border-radius: 12px !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 12px rgba(0, 123, 255, 0.3) !important;
        transition: all 0.3s ease !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(0, 123, 255, 0.4) !important;
        background: linear-gradient(135deg, #0056b3 0%, #003d82 100%) !important;
    }

    .stButton > button:active {
        transform: translateY(0px) !important;
    }

    .stButton > button:disabled {
        background: #6c757d !important;
        color: white !important;
        opacity: 0.6 !important;
        cursor: not-allowed !important;
        transform: none !important;
        box-shadow: none !important;
    }

    /* Tab styling - Enhanced */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 12px;
        border: 1px solid #dee2e6;
    }

    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%) !important;
        border-radius: 10px !important;
        color: #495057 !important;
        font-weight: 600 !important;
        border: 2px solid #dee2e6 !important;
        padding: 0.75rem 1.5rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
    }

    .stTabs [data-baseweb="tab"]:hover {
        border-color: #007bff !important;
        background: linear-gradient(135deg, #e7f3ff 0%, #cce7ff 100%) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 2px 6px rgba(0,123,255,0.2) !important;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #007bff 0%, #0056b3 100%) !important;
        color: white !important;
        border-color: #0056b3 !important;
        box-shadow: 0 3px 8px rgba(0,123,255,0.3) !important;
    }

    /* Subheader styling */
    .css-10trblm {
        color: #212529 !important;
    }

    /* Checkbox styling - Fixed for proper visibility */
    .stCheckbox {
        margin-bottom: 0.5rem;
    }

    /* Checkbox container */
    .stCheckbox > label {
        color: #212529 !important;
        background: transparent !important;
        padding: 0.6rem 1rem !important;
        border-radius: 0px !important;
        border: none !important;
        margin-bottom: 0.4rem !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
        cursor: pointer !important;
        box-shadow: none !important;
        display: flex !important;
        align-items: center !important;
    }

    /* Checkbox input - Force white background when unchecked */
    .stCheckbox input[type="checkbox"] {
        appearance: none !important;
        -webkit-appearance: none !important;
        -moz-appearance: none !important;
        width: 18px !important;
        height: 18px !important;
        border: 2px solid #6c757d !important;
        border-radius: 3px !important;
        background-color: #ffffff !important;
        background: #ffffff !important;
        margin-right: 0.5rem !important;
        position: relative !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        outline: none !important;
    }

    /* Force unchecked state to be white with border */
    .stCheckbox input[type="checkbox"]:not(:checked) {
        background-color: #ffffff !important;
        background: #ffffff !important;
        background-image: none !important;
        border: 2px solid #6c757d !important;
    }

    /* Checkbox checked state - Blue background with white checkmark */
    .stCheckbox input[type="checkbox"]:checked {
        background-color: #007bff !important;
        background: #007bff !important;
        border-color: #0056b3 !important;
    }

    /* Checkmark icon */
    .stCheckbox input[type="checkbox"]:checked::after {
        content: "✓" !important;
        color: white !important;
        font-size: 12px !important;
        font-weight: bold !important;
        position: absolute !important;
        top: 50% !important;
        left: 50% !important;
        transform: translate(-50%, -50%) !important;
    }

    /* Label hover effect */
    .stCheckbox > label:hover {
        background: transparent !important;
        transform: none !important;
        box-shadow: none !important;
    }

    /* Selected checkbox label styling */
    .stCheckbox:has(input:checked) > label {
        background: transparent !important;
        box-shadow: none !important;
    }

    /* Simple selectbox styling */
    .stSelectbox {
        margin-bottom: 1rem;
    }

    /* Target Streamlit's specific checkbox elements */
    .stCheckbox div[data-testid="stCheckbox"] input {
        background: #ffffff !important;
        background-color: #ffffff !important;
        border: 2px solid #6c757d !important;
        appearance: none !important;
        -webkit-appearance: none !important;
    }

    /* Target Streamlit's checkbox widget directly */
    div[data-testid="stCheckbox"] > label {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0.2rem 0 !important;
    }

    /* Remove any background from checkbox containers */
    div[data-testid="stCheckbox"] > label > div {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }

    /* Target the actual checkbox input element */
    div[data-testid="stCheckbox"] input[type="checkbox"] {
        background: #ffffff !important;
        background-color: #ffffff !important;
        background-image: none !important;
        border: 1px solid #6c757d !important;
        appearance: none !important;
        -webkit-appearance: none !important;
        width: 16px !important;
        height: 16px !important;
        margin-right: 0.5rem !important;
    }

    /* Additional override for unchecked state - most specific */
    .stCheckbox input[type="checkbox"]:not(:checked) {
        background-image: none !important;
        background-color: #ffffff !important;
        background: #ffffff !important;
    }
    }

    /* Number input styling - Enhanced */
    .stNumberInput > div > div {
        background-color: #ffffff !important;
        border: 2px solid #dee2e6 !important;
        border-radius: 10px !important;
        color: #212529 !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
        padding: 0.5rem 1rem !important;
        min-height: 48px !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }

    .stNumberInput > div > div:hover {
        border-color: #007bff !important;
        box-shadow: 0 3px 8px rgba(0,123,255,0.2) !important;
        transform: translateY(-1px) !important;
    }

    .stNumberInput > div > div:focus-within {
        border-color: #007bff !important;
        box-shadow: 0 0 0 4px rgba(0,123,255,0.15) !important;
    }

    /* Number input label */
    .stNumberInput > label {
        color: #212529 !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        margin-bottom: 0.5rem !important;
    }

    /* Number input field */
    .stNumberInput input {
        color: #212529 !important;
        font-weight: 500 !important;
        border: none !important;
        background-color: transparent !important;
    }

    .stNumberInput input:focus {
        outline: none !important;
    }

    /* Caption text */
    .caption {
        color: #6c757d !important;
        font-size: 0.875rem;
    }

    /* Dataframe styling */
    .stDataFrame {
        border: 1px solid #dee2e6;
        border-radius: 8px;
    }

    /* Additional sidebar fixes for different Streamlit versions */
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa !important;
    }

    section[data-testid="stSidebar"] .stMarkdown {
        color: #212529 !important;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #2E86AB !important;
    }

    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] li,
    section[data-testid="stSidebar"] span {
        color: #212529 !important;
    }

    /* Sidebar metric values */
    section[data-testid="stSidebar"] [data-testid="metric-container"] {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%) !important;
        border: 1px solid #dee2e6 !important;
        border-radius: 8px !important;
        padding: 1rem !important;
        margin: 0.5rem 0 !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
    }

    section[data-testid="stSidebar"] [data-testid="metric-container"] div {
        color: #212529 !important;
    }

    /* Sidebar scrollbar */
    section[data-testid="stSidebar"]::-webkit-scrollbar {
        width: 8px;
    }

    section[data-testid="stSidebar"]::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 4px;
    }

    section[data-testid="stSidebar"]::-webkit-scrollbar-thumb {
        background: #c1c1c1;
        border-radius: 4px;
    }

    section[data-testid="stSidebar"]::-webkit-scrollbar-thumb:hover {
        background: #a8a8a8;
    }

    /* Slider styling */
    .stSlider > div > div > div > div {
        background-color: #007bff !important;
    }

    .stSlider > div > div > div {
        background-color: #e9ecef !important;
    }

    /* Text input styling */
    .stTextInput > div > div {
        background-color: #ffffff !important;
        border: 2px solid #dee2e6 !important;
        border-radius: 8px !important;
        color: #212529 !important;
        transition: all 0.3s ease !important;
    }

    .stTextInput > div > div:focus-within {
        border-color: #007bff !important;
        box-shadow: 0 0 0 3px rgba(0,123,255,0.1) !important;
    }

    /* Radio button styling */
    .stRadio > div {
        background-color: #f8f9fa !important;
        border-radius: 8px !important;
        padding: 1rem !important;
        border: 1px solid #dee2e6 !important;
    }

    .stRadio label {
        color: #212529 !important;
        font-weight: 500 !important;
    }

    /* Date input styling */
    .stDateInput > div > div {
        background-color: #ffffff !important;
        border: 2px solid #dee2e6 !important;
        border-radius: 8px !important;
        color: #212529 !important;
    }

    /* File uploader styling */
    .stFileUploader > div {
        background-color: #f8f9fa !important;
        border: 2px dashed #007bff !important;
        border-radius: 12px !important;
        padding: 2rem !important;
        text-align: center !important;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #f8f9fa !important;
        border: 1px solid #dee2e6 !important;
        border-radius: 8px !important;
        color: #212529 !important;
    }

    /* Progress bar styling */
    .stProgress > div > div > div {
        background-color: #007bff !important;
    }

    /* Multiselect styling */
    .stMultiSelect > div > div {
        background-color: #ffffff !important;
        border: 2px solid #dee2e6 !important;
        border-radius: 8px !important;
    }

    /* Column styling for better spacing */
    .css-1kyxreq {
        padding: 0.5rem !important;
    }

    /* Container styling */
    .css-1y4p8pa {
        padding: 2rem 1rem !important;
    }

    /* Remove default Streamlit margins */
    .css-1v3fvcr {
        padding-top: 1rem !important;
    }

    /* Custom spacing for sections */
    .section-divider {
        height: 2px;
        background: linear-gradient(90deg, #007bff, #0056b3);
        margin: 2rem 0;
        border-radius: 1px;
    }

    /* Additional aggressive selectbox styling to ensure visibility */
    [data-baseweb="select"] {
        background-color: #ffffff !important;
    }

    [data-baseweb="select"] * {
        background-color: #ffffff !important;
        color: #212529 !important;
    }

    [data-baseweb="popover"] {
        background-color: #ffffff !important;
    }

    [data-baseweb="popover"] * {
        background-color: #ffffff !important;
        color: #212529 !important;
    }

    /* Target all possible select-related elements */
    .stSelectbox * {
        background-color: #ffffff !important;
        color: #212529 !important;
    }

    /* Override any dark themes */
    .stSelectbox .css-1n76uvr,
    .stSelectbox .css-1wy0on6,
    .stSelectbox .css-26l3qy-menu,
    .stSelectbox .css-4ljt47-MenuList {
        background-color: #ffffff !important;
        color: #212529 !important;
        border: 1px solid #dee2e6 !important;
    }

    /* Age number input additional styling */
    .stNumberInput * {
        background-color: #ffffff !important;
        color: #212529 !important;
    }

    /* Specific styling for form sections */
    .stNumberInput[data-testid="stNumberInput"] {
        background-color: #f8f9fa !important;
        padding: 1rem !important;
        border-radius: 10px !important;
        border: 1px solid #dee2e6 !important;
        margin: 0.5rem 0 !important;
    }

    .stSelectbox[data-testid="stSelectbox"] {
        background-color: #f8f9fa !important;
        padding: 1rem !important;
        border-radius: 10px !important;
        border: 1px solid #dee2e6 !important;
        margin: 0.5rem 0 !important;
    }

    /* Form section headers */
    h4 {
        color: #495057 !important;
        font-weight: 600 !important;
        margin-top: 1.5rem !important;
        margin-bottom: 0.5rem !important;
        padding: 0.5rem 0 !important;
        border-bottom: 1px solid #dee2e6 !important;
    }

    /* Help text styling */
    .stSelectbox .css-1cpxqw2, .stNumberInput .css-1cpxqw2 {
        color: #6c757d !important;
        font-size: 0.875rem !important;
        margin-top: 0.25rem !important;
    }

    /* Enhanced form sections with better spacing */
    .form-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1.5rem 0;
        border: 1px solid #dee2e6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    /* Improved spacing for form elements */
    .stSelectbox > div > div {
        margin-bottom: 1rem !important;
    }

    .stNumberInput > div {
        margin-bottom: 1rem !important;
    }

    /* Enhanced section separators */
    .element-container:has(.stSubheader) {
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
    }



    /* Ensure form sections take full width within their container */
    .stColumns > div {
        min-width: 0 !important;
        flex: 1 1 auto !important;
    }



    /* Better visual hierarchy for markdown headers */
    h3 {
        color: #2c3e50 !important;
        font-weight: 600 !important;
        margin-top: 1.5rem !important;
        margin-bottom: 1rem !important;
        border-bottom: 2px solid #3498db !important;
        padding-bottom: 0.5rem !important;
        display: block !important;
    }

    /* Section spacing improvements */
    .stMarkdown > div > h3 {
        margin-top: 2rem !important;
        margin-bottom: 1.5rem !important;
    }

    /* Better column alignment in forms */
    .stColumns > div {
        padding: 0 0.5rem !important;
    }

    /* Enhanced visual separation between sections */
    .stSubheader {
        margin-top: 2.5rem !important;
        margin-bottom: 1.5rem !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_info():
    """Load the trained model and metadata"""
    try:
        # Check if model files exist
        model_path = 'models/best_employment_model.joblib'
        info_path = 'models/model_info.json'

        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")
            return None, None

        if not os.path.exists(info_path):
            st.error(f"Model info file not found: {info_path}")
            return None, None

        model = joblib.load(model_path)
        with open(info_path, 'r') as f:
            model_info = json.load(f)
        return model, model_info
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.write("Please make sure you have:")
        st.write("1. Run the model training notebook")
        st.write("2. The 'models' directory exists with trained model files")
        return None, None

# Define skill families (same as in preprocessing)
SKILL_FAMILIES = {
    'Programming': [
        'Python', 'Java', 'JavaScript', 'C++', 'C#', 'C', 'PHP', 'Ruby',
        'Go', 'Rust', 'Swift', 'Kotlin', 'Scala', 'R', 'Matlab', 'Perl',
        'TypeScript', 'Dart', 'F#', 'Assembly', 'Delphi', 'VBA'
    ],

    'Web': [
        'HTML/CSS', 'React.js', 'Angular', 'Vue.js', 'Node.js', 'Express',
        'jQuery', 'Angular.js', 'Svelte', 'Django', 'Flask', 'Laravel',
        'Ruby on Rails', 'ASP.NET', 'ASP.NET Core', 'Spring', 'FastAPI'
    ],

    'Database': [
        'MySQL', 'PostgreSQL', 'MongoDB', 'SQLite', 'Redis', 'Oracle',
        'Microsoft SQL Server', 'MariaDB', 'DynamoDB', 'Elasticsearch',
        'Couchbase', 'Firebase', 'SQL'
    ],

    'CloudDevOps': [
        'AWS', 'Microsoft Azure', 'Google Cloud Platform', 'Docker',
        'Kubernetes', 'Git', 'Terraform', 'Ansible', 'Heroku',
        'DigitalOcean', 'Bash/Shell', 'PowerShell'
    ]
}

def calculate_skill_scores(selected_skills):
    """Calculate skill family scores based on selected skills (same logic as preprocessing)"""
    scores = {}
    binary_flags = {}

    for family_name, family_techs in SKILL_FAMILIES.items():
        # Count selected technologies from this family
        selected_from_family = [skill for skill in selected_skills if skill in family_techs]
        selected_count = len(selected_from_family)
        total_count = len(family_techs)

        # Calculate percentage score
        percentage = (selected_count / total_count) * 100 if total_count > 0 else 0

        # Store results
        scores[f'{family_name}_Score'] = round(percentage, 1)
        binary_flags[f'Has_{family_name}'] = 1 if selected_count > 0 else 0

    return scores, binary_flags

def calculate_derived_features(binary_flags):
    """Calculate skill breadth and full-stack indicator (same logic as preprocessing)"""
    skill_breadth = sum(binary_flags.values())

    is_fullstack = 1 if (binary_flags['Has_Programming'] == 1 and
                        binary_flags['Has_Web'] == 1 and
                        binary_flags['Has_Database'] == 1) else 0

    return {
        'Skill_Breadth': skill_breadth,
        'Is_FullStack': is_fullstack
    }

def create_feature_vector(user_inputs, feature_names):
    """Create feature vector from user inputs using same preprocessing logic"""
    # Initialize with zeros
    features = pd.DataFrame(0, index=[0], columns=feature_names)

    # Calculate skill scores and flags from selected skills
    scores, binary_flags = calculate_skill_scores(user_inputs['selected_skills'])
    derived_features = calculate_derived_features(binary_flags)

    # Set skill scores
    for score_name, score_value in scores.items():
        if score_name in feature_names:
            features.loc[0, score_name] = score_value

    # Set binary flags
    for flag_name, flag_value in binary_flags.items():
        if flag_name in feature_names:
            features.loc[0, flag_name] = flag_value

    # Set derived features
    for derived_name, derived_value in derived_features.items():
        if derived_name in feature_names:
            features.loc[0, derived_name] = derived_value

    # CRITICAL FIX: Use actual count of selected skills for ComputerSkills
    # This matches the preprocessing logic: ComputerSkills = total count of technologies known
    if 'ComputerSkills' in feature_names:
        skill_count = len(user_inputs['selected_skills'])

        # Breakdown by skill family for debugging
        prog_count = len([s for s in user_inputs['selected_skills'] if s in SKILL_FAMILIES['Programming']])
        web_count = len([s for s in user_inputs['selected_skills'] if s in SKILL_FAMILIES['Web']])
        db_count = len([s for s in user_inputs['selected_skills'] if s in SKILL_FAMILIES['Database']])
        cloud_count = len([s for s in user_inputs['selected_skills'] if s in SKILL_FAMILIES['CloudDevOps']])

        # Cap at reasonable maximum (training data shows max ~30-40 skills)
        features.loc[0, 'ComputerSkills'] = min(skill_count, 40)
        print(f"[DEBUG] ComputerSkills = {skill_count} (Prog:{prog_count} + Web:{web_count} + DB:{db_count} + Cloud:{cloud_count})")

    if 'EducationLevel_Numeric' in feature_names:
        features.loc[0, 'EducationLevel_Numeric'] = user_inputs['education_level']
    if 'IsYoung' in feature_names:
        features.loc[0, 'IsYoung'] = 1 if user_inputs['age'] < 35 else 0
    if 'IsDeveloper' in feature_names:
        features.loc[0, 'IsDeveloper'] = 1 if user_inputs['is_developer'] else 0
    if 'HasMentalHealthConcerns' in feature_names:
        features.loc[0, 'HasMentalHealthConcerns'] = 1 if user_inputs['mental_health'] else 0
    if 'HasAccessibilityNeeds' in feature_names:
        features.loc[0, 'HasAccessibilityNeeds'] = 1 if user_inputs['accessibility'] else 0

    # Gender (one-hot encoded)
    if user_inputs['gender'] == 'Man' and 'Gender_Man' in feature_names:
        features.loc[0, 'Gender_Man'] = 1
    elif user_inputs['gender'] == 'Woman' and 'Gender_Woman' in feature_names:
        features.loc[0, 'Gender_Woman'] = 1
    elif 'Gender_NonBinary' in feature_names:
        features.loc[0, 'Gender_NonBinary'] = 1

    if 'HasProfessionalExperience' in feature_names:
        features.loc[0, 'HasProfessionalExperience'] = 1 if user_inputs['prof_experience'] else 0
    if 'HasSalaryInfo' in feature_names:
        features.loc[0, 'HasSalaryInfo'] = 1 if user_inputs['salary_info'] else 0

    return features

def reality_check_prediction(probability, user_inputs, scores, derived):
    """Apply reality check to predictions based on employment market patterns"""
    skill_count = len(user_inputs['selected_skills'])

    # Define employment likelihood adjustment factors
    adjustment_factors = []

    # Critical factors that strongly impact employment
    if skill_count <= 2:
        adjustment_factors.append(("Very few skills", -0.3))
    elif skill_count <= 5:
        adjustment_factors.append(("Limited skills", -0.15))

    if not user_inputs['prof_experience']:
        adjustment_factors.append(("No professional experience", -0.2))

    if not user_inputs['is_developer']:
        adjustment_factors.append(("Not in developer role", -0.15))

    if scores['Programming_Score'] < 15:
        adjustment_factors.append(("Very limited programming", -0.2))

    # Positive factors
    if derived['Is_FullStack']:
        adjustment_factors.append(("Full-stack capability", +0.1))

    if skill_count >= 15:
        adjustment_factors.append(("Extensive skills", +0.1))

    # Calculate adjusted probability
    total_adjustment = sum(factor[1] for factor in adjustment_factors)
    adjusted_probability = max(0.05, min(0.95, probability + total_adjustment))

    return adjusted_probability, adjustment_factors

def get_recommendations(probability, user_inputs):
    """Generate personalized recommendations"""
    recommendations = []

    # Calculate skill scores for recommendations
    scores, binary_flags = calculate_skill_scores(user_inputs['selected_skills'])
    skill_count = len(user_inputs['selected_skills'])

    # Priority recommendations based on critical gaps
    if skill_count <= 3:
        recommendations.append("� **CRITICAL:** Learn at least 5-8 core technologies")
        recommendations.append("📚 Focus on popular languages: Python, JavaScript, or Java")

    if not user_inputs['prof_experience'] and not user_inputs['is_developer']:
        recommendations.append("💼 **URGENT:** Gain practical experience through projects or internships")

    # Skill-specific recommendations
    if scores['Programming_Score'] < 30:
        recommendations.append("� Master at least 2-3 programming languages")
    if scores['Web_Score'] < 20 and scores['Programming_Score'] > 30:
        recommendations.append("� Add web development skills (HTML/CSS, React, Node.js)")
    if scores['Database_Score'] < 15:
        recommendations.append("🗄️ Learn database fundamentals (SQL, MongoDB)")
    if scores['CloudDevOps_Score'] < 15 and skill_count > 8:
        recommendations.append("☁️ Add cloud/DevOps skills for senior roles")

    # General recommendations based on probability ranges
    if probability < 0.4:
        recommendations.append("🎯 Focus on building a strong foundation before applying")
        recommendations.append("🏫 Consider bootcamps or formal education")
    elif probability < 0.6:
        recommendations.append("📈 You're on the right track - keep learning!")
        recommendations.append("🤝 Network with developers in your areas of interest")
    elif probability < 0.8:
        recommendations.append("� Start applying to positions matching your skills")
        recommendations.append("� Highlight your technical strengths in applications")
    else:
        recommendations.append("🎯 You have strong employability - apply confidently!")
        recommendations.append("� Consider specialized or senior positions")

    return recommendations[:5]  # Return top 5 recommendations

def main():
    # Header
    st.markdown('<h1 class="main-header">PredSeeker</h1>', unsafe_allow_html=True)
    st.markdown('<h3 class="sub-header">AI-Powered Developer Employment Predictor</h3>', unsafe_allow_html=True)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Load model
    model, model_info = load_model_and_info()

    if model is None:
        st.error("❌ Could not load the trained model. Please check the model files.")
        st.info("💡 Make sure you have run the model training notebook and the model files exist in the 'models' directory.")
        return

    # Sidebar - Model Info with enhanced styling
    with st.sidebar:
        # Model Performance Section
        st.markdown("""
        <div style="background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
                   border: 1px solid #dee2e6; border-radius: 10px; padding: 1.5rem; margin: 1rem 0;">
            <h3 style="color: #2E86AB; text-align: center; margin-bottom: 1rem; border-bottom: 2px solid #dee2e6; padding-bottom: 0.5rem;">
                📊 Model Performance
            </h3>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("ROC-AUC", f"{model_info['performance_metrics']['ROC-AUC']:.3f}")
            st.metric("Accuracy", f"{model_info['performance_metrics']['Accuracy']:.3f}")
        with col2:
            st.metric("F1-Score", f"{model_info['performance_metrics']['F1-Score']:.3f}")
            st.metric("Precision", f"{model_info['performance_metrics']['Precision']:.3f}")

        st.markdown("</div>", unsafe_allow_html=True)

        # Model Info Section
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
                   border: 1px solid #dee2e6; border-radius: 10px; padding: 1.5rem; margin: 1rem 0;">
            <h3 style="color: #2E86AB; text-align: center; margin-bottom: 1rem; border-bottom: 2px solid #dee2e6; padding-bottom: 0.5rem;">
                🤖 Model Info
            </h3>
            <div style="color: #212529; line-height: 1.6;">
                <p><strong>Algorithm:</strong> {model_info['model_name']}</p>
                <p><strong>Features:</strong> {len(model_info['features'])}</p>
                <p><strong>Training Date:</strong> {model_info['training_date']}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Dataset Stats Section
        st.markdown("""
        <div style="background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
                   border: 1px solid #dee2e6; border-radius: 10px; padding: 1.5rem; margin: 1rem 0;">
            <h3 style="color: #2E86AB; text-align: center; margin-bottom: 1rem; border-bottom: 2px solid #dee2e6; padding-bottom: 0.5rem;">
                📈 Dataset Stats
            </h3>
            <div style="color: #212529; line-height: 1.6;">
                <p><strong>Training Samples:</strong> 58,769</p>
                <p><strong>Test Samples:</strong> 14,693</p>
                <p><strong>Data Source:</strong> Stack Overflow Survey</p>
                <p><strong>Employment Rate:</strong> ~74% (Balanced Dataset)</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Additional Info
        st.markdown("""
        <div style="background: linear-gradient(135deg, #cce7ff 0%, #b3d7ff 100%);
                   border: 1px solid #007bff; border-radius: 10px; padding: 1rem; margin: 1rem 0; text-align: center;">
            <p style="color: #004085; margin: 0; font-size: 0.9rem;">
                <strong>💡 Tip:</strong> Select diverse skills across multiple categories for better employment predictions!
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Main content
    tab1, tab2 = st.tabs(["🔮 Make Prediction", "📊 About the Model"])

    with tab1:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.header("📝 Enter Your Developer Profile")

            # Skills Selection Section
            st.subheader("💻 Technical Skills - Select Your Skills")

            # Create tabs for each skill family
            tab_prog, tab_web, tab_db, tab_cloud = st.tabs(["Programming", "Web Tech", "Database", "Cloud/DevOps"])

            selected_skills = []

            with tab_prog:
                #st.markdown('<div class="skill-category">', unsafe_allow_html=True)
                st.write("**💻 Programming Languages & Frameworks**")
                #st.markdown('</div>', unsafe_allow_html=True)
                prog_cols = st.columns(3)
                for i, skill in enumerate(SKILL_FAMILIES['Programming']):
                    with prog_cols[i % 3]:
                        if st.checkbox(skill, key=f"prog_{skill}"):
                            selected_skills.append(skill)

            with tab_web:
                #st.markdown('<div class="skill-category">', unsafe_allow_html=True)
                st.write("**🌐 Web Development Technologies**")
                #st.markdown('</div>', unsafe_allow_html=True)
                web_cols = st.columns(3)
                for i, skill in enumerate(SKILL_FAMILIES['Web']):
                    with web_cols[i % 3]:
                        if st.checkbox(skill, key=f"web_{skill}"):
                            selected_skills.append(skill)

            with tab_db:
                #st.markdown('<div class="skill-category">', unsafe_allow_html=True)
                st.write("**🗄️ Database & Data Storage**")
                #st.markdown('</div>', unsafe_allow_html=True)
                db_cols = st.columns(3)
                for i, skill in enumerate(SKILL_FAMILIES['Database']):
                    with db_cols[i % 3]:
                        if st.checkbox(skill, key=f"db_{skill}"):
                            selected_skills.append(skill)

            with tab_cloud:
                #st.markdown('<div class="skill-category">', unsafe_allow_html=True)
                st.write("**☁️ Cloud Computing & DevOps**")
                #st.markdown('</div>', unsafe_allow_html=True)
                cloud_cols = st.columns(3)
                for i, skill in enumerate(SKILL_FAMILIES['CloudDevOps']):
                    with cloud_cols[i % 3]:
                        if st.checkbox(skill, key=f"cloud_{skill}"):
                            selected_skills.append(skill)

            # Display selected skills summary
            if selected_skills:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
                           border-left: 4px solid #17a2b8; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
                    <strong style="color: #0c5460;">Selected {len(selected_skills)} skills:</strong>
                    <br><span style="color: #0c5460;">{', '.join(selected_skills[:5])}{' and ' + str(len(selected_skills) - 5) + ' more...' if len(selected_skills) > 5 else ''}</span>
                </div>
                """, unsafe_allow_html=True)

                # Calculate and display skill scores
                scores, binary_flags = calculate_skill_scores(selected_skills)
                derived = calculate_derived_features(binary_flags)

                st.markdown("### 📊 Your Skill Profile")
                score_cols = st.columns(4)
                with score_cols[0]:
                    st.metric("Programming", f"{scores['Programming_Score']:.1f}%",
                             f"{len([s for s in selected_skills if s in SKILL_FAMILIES['Programming']])}/{len(SKILL_FAMILIES['Programming'])}")
                with score_cols[1]:
                    st.metric("Web Tech", f"{scores['Web_Score']:.1f}%",
                             f"{len([s for s in selected_skills if s in SKILL_FAMILIES['Web']])}/{len(SKILL_FAMILIES['Web'])}")
                with score_cols[2]:
                    st.metric("Database", f"{scores['Database_Score']:.1f}%",
                             f"{len([s for s in selected_skills if s in SKILL_FAMILIES['Database']])}/{len(SKILL_FAMILIES['Database'])}")
                with score_cols[3]:
                    st.metric("Cloud/DevOps", f"{scores['CloudDevOps_Score']:.1f}%",
                             f"{len([s for s in selected_skills if s in SKILL_FAMILIES['CloudDevOps']])}/{len(SKILL_FAMILIES['CloudDevOps'])}")

                # Show special indicators with better styling
                if derived['Is_FullStack']:
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
                               border-left: 4px solid #28a745; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
                        <strong style="color: #155724;">🌟 Full-Stack Developer Profile Detected!</strong>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #cce7ff 0%, #b3d7ff 100%);
                           border-left: 4px solid #007bff; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
                    <strong style="color: #004085;">🎯 Skill Breadth: {derived['Skill_Breadth']} out of 4 skill categories</strong>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
                           border-left: 4px solid #ffc107; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
                    <strong style="color: #856404;">Please select at least one skill to continue</strong>
                </div>
                """, unsafe_allow_html=True)

            # Demographics Section with improved layout
            st.subheader("👤 Personal Information")

            # Age Section
            st.markdown("#### 📅 Age")
            age = st.number_input(
                "Enter your age",
                min_value=16,
                max_value=70,
                value=28,
                key="age_input",
                help="Your current age in years"
            )

            st.markdown("<br>", unsafe_allow_html=True)

            # Gender Section - Simple Layout
            st.markdown("#### 👥 Gender")
            gender = st.selectbox(
                "Select your gender:",
                ["Man", "Woman", "NonBinary"],
                index=0,
                key="gender_select"
            )

            # Professional Section with better spacing
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("💼 Professional Background")

            # Calculate computer skills automatically from selected skills
            total_selected_skills = len(selected_skills)

            # Display computer skills as calculated value (not editable)
            if total_selected_skills > 0:
                # Calculate breakdown for display
                prog_selected = len([s for s in selected_skills if s in SKILL_FAMILIES['Programming']])
                web_selected = len([s for s in selected_skills if s in SKILL_FAMILIES['Web']])
                db_selected = len([s for s in selected_skills if s in SKILL_FAMILIES['Database']])
                cloud_selected = len([s for s in selected_skills if s in SKILL_FAMILIES['CloudDevOps']])

                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
                           border-left: 4px solid #17a2b8; padding: 1.5rem; border-radius: 10px; margin: 1rem 0; text-align: center;">
                    <h3 style="color: #0c5460; margin-bottom: 0.5rem;">🖥️ Computer Skills Count: {total_selected_skills}</h3>
                    <p style="color: #0c5460; margin: 0; font-size: 1rem;">
                        <strong>Auto-calculated Breakdown:</strong><br>
                        Programming({prog_selected}) + Web({web_selected}) + Database({db_selected}) + Cloud/DevOps({cloud_selected}) = {total_selected_skills} total skills
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #cce7ff 0%, #b3d7ff 100%);
                           border-left: 4px solid #007bff; padding: 1.5rem; border-radius: 10px; margin: 1rem 0; text-align: center;">
                    <h3 style="color: #004085; margin-bottom: 0.5rem;">🖥️ Computer Skills Count: 0</h3>
                    <p style="color: #004085; margin: 0;">Select skills above to update automatically</p>
                </div>
                """, unsafe_allow_html=True)

            # Education Level Section - Simple Layout
            st.markdown("#### 🎓 Education Level")

            education_options = [
                'No Higher Education',
                'Other/Some College',
                'Undergraduate/Bachelor\'s',
                'Master\'s Degree',
                'PhD/Doctorate'
            ]

            education_mapping = {
                'No Higher Education': 0,
                'Other/Some College': 1,
                'Undergraduate/Bachelor\'s': 2,
                'Master\'s Degree': 3,
                'PhD/Doctorate': 4
            }

            education_display = st.selectbox(
                "Select your highest education level:",
                education_options,
                index=2,  # Default to Bachelor's
                key="education_select"
            )
            education_level = education_mapping[education_display]

            # Experience and Background
            st.subheader("🎯 Experience & Background")
            col1_exp, col2_exp = st.columns(2)
            with col1_exp:
                is_developer = st.checkbox("I am a professional developer", value=True)
                prof_experience = st.checkbox("I have professional coding experience", value=True)
            with col2_exp:
                mental_health = st.checkbox("I have mental health concerns")
                accessibility = st.checkbox("I have accessibility needs")

            salary_info = st.checkbox("I'm willing to share salary information")

            # Prediction Button (only show if skills are selected)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            if not selected_skills:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
                           border-left: 4px solid #ffc107; padding: 1rem; border-radius: 8px; margin: 1rem 0; text-align: center;">
                    <strong style="color: #856404;">⚠️ Please select at least one technical skill above to enable prediction</strong>
                </div>
                """, unsafe_allow_html=True)
                st.button("🔮 Predict Employment Probability", disabled=True, use_container_width=True)
            elif st.button("🔮 Predict Employment Probability", type="primary", use_container_width=True):
                user_inputs = {
                    'selected_skills': selected_skills,
                    'age': age,
                    'gender': gender,
                    'computer_skills': len(selected_skills),  # Auto-calculated from selected skills
                    'education_level': education_level,
                    'is_developer': is_developer,
                    'prof_experience': prof_experience,
                    'mental_health': mental_health,
                    'accessibility': accessibility,
                    'salary_info': salary_info
                }

                try:
                    # Create feature vector
                    features = create_feature_vector(user_inputs, model_info['features'])

                    # Calculate scores for internal use
                    skill_count = len(user_inputs['selected_skills'])
                    scores, binary_flags = calculate_skill_scores(user_inputs['selected_skills'])
                    derived = calculate_derived_features(binary_flags)

                    # Show brief skill summary
                    if skill_count < 3 and not user_inputs['prof_experience']:
                        st.warning("⚠️ **Note:** Limited skills and no professional experience may impact employability.")

                    # Make prediction with reality check
                    raw_prediction = model.predict(features)[0]
                    raw_probability = model.predict_proba(features)[0][1]

                    # Apply reality check (internal - no display)
                    adjusted_probability, _ = reality_check_prediction(
                        raw_probability, user_inputs, scores, derived
                    )

                    # Use adjusted prediction
                    prediction = 1 if adjusted_probability >= 0.5 else 0
                    probability = adjusted_probability

                    # Store results in session state
                    st.session_state['prediction'] = prediction
                    st.session_state['probability'] = probability
                    st.session_state['raw_probability'] = raw_probability
                    st.session_state['user_inputs'] = user_inputs
                    st.session_state['debug_info'] = {
                        'skill_count': skill_count,
                        'scores': scores,
                        'binary_flags': binary_flags,
                        'derived': derived
                    }

                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
                    st.text(f"Debug: Features shape: {features.shape if 'features' in locals() else 'Not created'}")
                    st.text(f"Debug: Model features expected: {len(model_info['features']) if model_info else 'Unknown'}")

        # Display results if prediction was made
        with col2:
            if 'prediction' in st.session_state:
                prediction = st.session_state['prediction']
                probability = st.session_state['probability']
                user_inputs = st.session_state['user_inputs']

                st.header("🎯 Prediction Result")

                # Result box with improved styling
                if prediction == 1:
                    st.markdown(f"""
                    <div class="prediction-box employed">
                        <h2>🎉 High Employment Probability</h2>
                        <h3>Success Rate: {probability:.1%}</h3>
                        <p style="font-size: 1.1rem; margin-top: 1rem;">Your technical profile shows strong employment potential!</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-box unemployed">
                        <h2>📈 Room for Improvement</h2>
                        <h3>Current Probability: {probability:.1%}</h3>
                        <p style="font-size: 1.1rem; margin-top: 1rem;">Consider expanding your technical skillset for better opportunities.</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Modern confidence gauge
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = probability * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Employment Probability", 'font': {'size': 16, 'color': '#2E86AB'}},
                    number = {'font': {'size': 24, 'color': '#2E86AB'}},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickcolor': '#6C757D'},
                        'bar': {'color': "#2E86AB", 'thickness': 0.8},
                        'steps': [
                            {'range': [0, 40], 'color': "#FFE5E5"},
                            {'range': [40, 70], 'color': "#FFF5E5"},
                            {'range': [70, 100], 'color': "#E5F5E5"}
                        ],
                        'threshold': {
                            'line': {'color': "#DC3545", 'width': 3},
                            'thickness': 0.8,
                            'value': 50
                        },
                        'bordercolor': "#E9ECEF",
                        'borderwidth': 2
                    }
                ))
                fig.update_layout(
                    height=280,
                    margin=dict(l=20, r=20, t=50, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={'family': 'Arial, sans-serif'}
                )
                st.plotly_chart(fig, use_container_width=True)

                # Skills radar chart
                scores, _ = calculate_skill_scores(user_inputs['selected_skills'])
                skills_data = {
                    'Programming': scores['Programming_Score'],
                    'Web Tech': scores['Web_Score'],
                    'Database': scores['Database_Score'],
                    'Cloud/DevOps': scores['CloudDevOps_Score']
                }

                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=list(skills_data.values()),
                    theta=list(skills_data.keys()),
                    fill='toself',
                    name='Your Skills',
                    line=dict(color='#2E86AB', width=3),
                    fillcolor='rgba(46, 134, 171, 0.3)',
                    marker=dict(color='#2E86AB', size=8)
                ))
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 100],
                            tickfont=dict(size=10, color='#6C757D'),
                            gridcolor='#E9ECEF'
                        ),
                        angularaxis=dict(
                            tickfont=dict(size=12, color='#495057'),
                            gridcolor='#E9ECEF'
                        ),
                        bgcolor='rgba(0,0,0,0)'
                    ),
                    showlegend=False,
                    title=dict(text="Skills Profile", x=0.5, font=dict(size=16, color='#2E86AB')),
                    height=280,
                    margin=dict(l=20, r=20, t=50, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={'family': 'Arial, sans-serif'}
                )
                st.plotly_chart(fig_radar, use_container_width=True)

                # Recommendations
                st.subheader("💡 Recommendations")
                recommendations = get_recommendations(probability, user_inputs)
                for rec in recommendations:
                    st.write(f"• {rec}")


    with tab2:
        st.header("📊 About the Employment Prediction Model")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🎯 How It Works")
            st.write("""
            This AI model predicts developer employment probability using:

            • **Machine Learning Algorithm:** XGBoost Gradient Boosting
            • **Training Data:** 73,462 developers from Stack Overflow Survey
            • **Features:** 21 engineered features including skills, demographics, and experience
            • **Accuracy:** 89.5% ROC-AUC score on test data

            The model analyzes your technical skills, professional background, and demographic information to provide employment insights.
            """)

            st.subheader("📈 Key Features")
            st.write("""
            • **Skill Scores:** Programming, Web, Database, Cloud/DevOps
            • **Derived Metrics:** Skill breadth, full-stack capability
            • **Demographics:** Age, gender, accessibility needs
            • **Professional:** Experience, education, developer status
            """)

        with col2:
            st.subheader("🏆 Model Performance")

            # Performance metrics
            metrics_data = {
                'Metric': ['ROC-AUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score'],
                'Score': [
                    model_info['performance_metrics']['ROC-AUC'],
                    model_info['performance_metrics']['Accuracy'],
                    model_info['performance_metrics']['Precision'],
                    model_info['performance_metrics']['Recall'],
                    model_info['performance_metrics']['F1-Score']
                ]
            }

            fig_metrics = px.bar(
                x=metrics_data['Score'],
                y=metrics_data['Metric'],
                orientation='h',
                title="Model Performance Metrics",
                color=metrics_data['Score'],
                color_continuous_scale=['#FFE5E5', '#2E86AB'],
                text=[f"{score:.3f}" for score in metrics_data['Score']]
            )
            fig_metrics.update_layout(
                height=320,
                showlegend=False,
                title=dict(x=0.5, font=dict(size=16, color='#2E86AB')),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=50, b=20),
                font={'family': 'Arial, sans-serif'},
                xaxis=dict(gridcolor='#E9ECEF', tickfont=dict(color='#6C757D')),
                yaxis=dict(gridcolor='#E9ECEF', tickfont=dict(color='#6C757D'))
            )
            fig_metrics.update_traces(textposition='outside', textfont=dict(color='#495057'))
            st.plotly_chart(fig_metrics, use_container_width=True)

            st.subheader("⚠️ Limitations")
            st.write("""
            • Predictions are probabilistic, not guarantees
            • Based on survey data from specific time period
            • Individual circumstances may vary
            • Use as guidance, not definitive career advice
            """)

    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; color: #6C757D; padding: 30px; background-color: #f8f9fa; border-radius: 10px; margin-top: 2rem;">
        <p style="font-size: 1.2rem; margin-bottom: 1rem;"><strong>🎯 PredSeeker</strong> - AI-Powered Developer Employment Prediction</p>
        <p style="margin-bottom: 0.5rem;">Built with ❤️ using Streamlit • Powered by XGBoost ML Algorithm</p>
        <p style="font-size: 0.9rem; color: #868e96; margin-bottom: 0;"><em>Disclaimer: This tool provides probabilistic predictions for guidance only. Individual results may vary.</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
