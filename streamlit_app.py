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
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with better font family and improved spacing
st.markdown("""
<style>
    /* Import professional fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global font family and base styling */
    html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', 'Roboto', sans-serif !important;
        color: #1f2937 !important;
    }
    
    /* Ensure all text elements have dark colors */
    p, span, div, label {
        color: #1f2937 !important;
    }
    
    .stMarkdown p {
        color: #1f2937 !important;
    }
    
    /* Main app styling */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        color: #2c3e50;
        min-height: 100vh;
    }
    
    /* Main container with compact spacing */
    .main .block-container {
        max-width: 1400px;
        padding: 1.5rem 1rem;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        margin: 1rem auto;
    }
    
    /* Remove default Streamlit styling */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Enhanced header styling */
    .main-header {
        font-family: 'Poppins', sans-serif !important;
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -2px;
        line-height: 1.1;
    }
    
    .sub-header {
        font-family: 'Inter', sans-serif !important;
        font-size: 1.2rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
        letter-spacing: 0.5px;
    }
    
    /* Enhanced prediction result area */
    .prediction-container {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 16px;
        padding: 1.5rem 1rem;
        margin: 1.5rem 0;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .prediction-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 6px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 24px 24px 0 0;
    }
    
    .prediction-result-header {
        font-family: 'Poppins', sans-serif !important;
        font-size: 2rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 1rem;
        color: #2c3e50;
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border-radius: 16px;
        padding: 1.5rem 1rem;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.08);
        border: 2px solid transparent;
        position: relative;
        transition: all 0.3s ease;
    }
    
    .prediction-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.12);
    }
    
    .employed {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border-color: #10b981;
        color: #064e3b !important;
    }
    
    .unemployed {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border-color: #ef4444;
        color: #7f1d1d !important;
    }
    
    .prediction-box h2 {
        font-family: 'Poppins', sans-serif !important;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
        color: inherit !important;
    }
    
    .prediction-box h3 {
        font-family: 'Inter', sans-serif !important;
        font-size: 1.8rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: inherit !important;
    }
    
    .prediction-box p {
        font-family: 'Inter', sans-serif !important;
        font-size: 1.2rem;
        font-weight: 400;
        line-height: 1.6;
        color: inherit !important;
        margin-top: 1rem;
    }
    
    /* Enhanced chart containers */
    .chart-container {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(0, 0, 0, 0.05);
    }
    
    .chart-title {
        font-family: 'Poppins', sans-serif !important;
        font-size: 1.4rem;
        font-weight: 600;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    /* Enhanced sidebar styling */
    .css-1d391kg, section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%) !important;
        border-right: 2px solid #e2e8f0 !important;
        padding: 2rem 1rem !important;
    }
    
    .sidebar-section {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(0, 0, 0, 0.05);
    }
    
    .sidebar-header {
        font-family: 'Poppins', sans-serif !important;
        font-size: 1.2rem;
        font-weight: 600;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 0.5rem;
    }
    
    /* Enhanced button styling */
    .stButton > button {
        font-family: 'Inter', sans-serif !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.7rem 2rem !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.3px !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.3) !important;
        transition: all 0.3s ease !important;
        text-transform: none !important;
        min-height: 45px !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 12px 35px rgba(102, 126, 234, 0.4) !important;
        background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%) !important;
    }
    
    .stButton > button:disabled {
        background: linear-gradient(135deg, #94a3b8 0%, #64748b 100%) !important;
        opacity: 0.6 !important;
        transform: none !important;
        box-shadow: none !important;
    }
    
    /* Enhanced tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        padding: 0.5rem;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-family: 'Inter', sans-serif !important;
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%) !important;
        border-radius: 12px !important;
        color: #1f2937 !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        border: 1px solid transparent !important;
        padding: 0.6rem 1.2rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05) !important;
        min-height: 40px !important;
    }
    
    .stTabs [data-baseweb="tab"] * {
        color: #1f2937 !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        border-color: #667eea !important;
        background: linear-gradient(135deg, #f0f4ff 0%, #e0e7ff 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.15) !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border-color: #5a67d8 !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.3) !important;
    }
    
    /* Section headers */
    h1, h2, h3 {
        font-family: 'Poppins', sans-serif !important;
        color: #2c3e50 !important;
        font-weight: 600 !important;
        margin-top: 1.5rem !important;
        margin-bottom: 1rem !important;
        line-height: 1.3 !important;
    }
    
    h1 {
        font-size: 2.2rem !important;
        border-bottom: 2px solid #667eea !important;
        padding-bottom: 0.5rem !important;
    }
    
    h2 {
        font-size: 1.8rem !important;
        border-bottom: 1px solid #e2e8f0 !important;
        padding-bottom: 0.5rem !important;
    }
    
    h3 {
        font-size: 1.4rem !important;
        color: #475569 !important;
    }
    
    /* Enhanced form styling */
    .stSelectbox, .stNumberInput, .stCheckbox {
        font-family: 'Inter', sans-serif !important;
        margin-bottom: 0.8rem !important;
        color: #1f2937 !important;
    }
    
    .stSelectbox label, .stNumberInput label, .stCheckbox label {
        font-family: 'Inter', sans-serif !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        color: #1f2937 !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Ensure all form text is dark */
    .stSelectbox *, .stNumberInput *, .stCheckbox * {
        color: #1f2937 !important;
    }
    
    /* Target the outer selectbox container */
    .stSelectbox > div {
        background: transparent !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    
    /* Target the actual selectbox element (remove nested card) */
    .stSelectbox > div > div {
        background: #ffffff !important;
        border: 2px solid #e2e8f0 !important;
        border-radius: 12px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 1.1rem !important;
        font-weight: 500 !important;
        color: #1f2937 !important;
        padding: 0.8rem 1.2rem !important;
        min-height: 50px !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08) !important;
        transition: all 0.3s ease !important;
        position: relative !important;
        display: flex !important;
        align-items: center !important;
    }
    
    /* Remove any inner card styling */
    .stSelectbox > div > div > div {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    
    .stSelectbox > div > div:hover {
        background: #ffffff !important;
        border-color: #667eea !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.15) !important;
    }
    
    .stSelectbox > div > div:focus-within {
        background: #ffffff !important;
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    }
    
    /* Fix selected value display in selectbox */
    .stSelectbox [data-baseweb="select"] {
        background: #ffffff !important;
        color: #1f2937 !important;
        border: none !important;
        padding: 0 !important;
        margin: 0 !important;
        box-shadow: none !important;
        border-radius: 0 !important;
        width: 100% !important;
        height: 100% !important;
        display: flex !important;
        align-items: center !important;
        font-size: 1.1rem !important;
        font-weight: 500 !important;
    }
    
    .stSelectbox [data-baseweb="select"] > div {
        color: #1f2937 !important;
        background: #ffffff !important;
        opacity: 1 !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        font-size: 1.1rem !important;
        font-weight: 500 !important;
        line-height: 1.4 !important;
    }
    
    .stSelectbox [data-baseweb="select"] span {
        color: #1f2937 !important;
        opacity: 1 !important;
        font-weight: 500 !important;
        padding-left: 0.2rem !important;
        margin-left: 0.2rem !important;
    }
    
    .stSelectbox [data-baseweb="select"] div[data-baseweb="input"] {
        color: #1f2937 !important;
        background: #ffffff !important;
        opacity: 1 !important;
        font-size: 1.1rem !important;
        font-weight: 500 !important;
        line-height: 1.4 !important;
        padding: 0 0 0 0.5rem !important;
        margin: 0 !important;
        border: none !important;
        width: 100% !important;
    }
    
    .stSelectbox [data-baseweb="select"] div[data-baseweb="input"] > div {
        color: #1f2937 !important;
        background: #ffffff !important;
        opacity: 1 !important;
        font-size: 1.1rem !important;
        font-weight: 500 !important;
        line-height: 1.4 !important;
        padding: 0 0 0 0.3rem !important;
        margin: 0 !important;
        border: none !important;
    }
    
    .stSelectbox span {
        color: #000000 !important;
        opacity: 1 !important;
    }
    
    .stSelectbox input {
        color: #000000 !important;
        background: #ffffff !important;
        opacity: 1 !important;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #667eea !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.15) !important;
    }
    
    /* Simple dropdown menu */
    .stSelectbox [role="listbox"] {
        background: #ffffff !important;
        border: 1px solid #d1d5db !important;
        border-radius: 6px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
        padding: 0.25rem 0 !important;
    }
    
    .stSelectbox [role="option"] {
        background: #ffffff !important;
        color: #1f2937 !important;
        padding: 0.5rem 1rem !important;
        font-size: 1rem !important;
    }
    
    .stSelectbox [role="option"]:hover {
        background: #f3f4f6 !important;
        color: #1f2937 !important;
    }
    
    .stSelectbox [role="option"][aria-selected="true"] {
        background: #3b82f6 !important;
        color: #ffffff !important;
    }
    
    /* Universal selectbox text visibility */
    .stSelectbox * {
        color: #1f2937 !important;
        opacity: 1 !important;
    }
    
    /* Additional specific targeting for selected value */
    .stSelectbox div[data-baseweb="select"] div {
        color: #1f2937 !important;
        opacity: 1 !important;
    }
    
    /* Ensure text in input area is visible */
    .stSelectbox > div > div input {
        color: #1f2937 !important;
        opacity: 1 !important;
        -webkit-text-fill-color: #1f2937 !important;
    }
    
    /* Target the actual displayed value */
    .stSelectbox div[role="combobox"] {
        color: #1f2937 !important;
        opacity: 1 !important;
    }
    
    .stSelectbox div[role="combobox"] * {
        color: #1f2937 !important;
        opacity: 1 !important;
    }
    
    /* Enhanced number input styling */
    .stNumberInput > div > div {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%) !important;
        border: 2px solid #e2e8f0 !important;
        border-radius: 12px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 1.1rem !important;
        font-weight: 500 !important;
        color: #1f2937 !important;
        padding: 0.8rem 1.2rem !important;
        min-height: 50px !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08) !important;
        border: 1px solid rgba(0, 0, 0, 0.05) !important;
        transition: all 0.3s ease !important;
    }
    
    /* Number input field */
    .stNumberInput input {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%) !important;
        color: #1f2937 !important;
        border: none !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 1.1rem !important;
        font-weight: 500 !important;
    }
    
    .stNumberInput input:focus {
        background-color: #ffffff !important;
        color: #1f2937 !important;
        outline: none !important;
    }
    
    /* Number input container overrides */
    .stNumberInput [data-baseweb="input"] {
        background-color: #ffffff !important;
        color: #1f2937 !important;
    }
    
    .stNumberInput [data-baseweb="input"] > div {
        background-color: #ffffff !important;
        color: #1f2937 !important;
    }
    
    /* Fix number input spinner buttons (increase/decrease) */
    .stNumberInput button {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%) !important;
        color: #1f2937 !important;
        border: none !important;
    }
    
    .stNumberInput button:hover {
        background: linear-gradient(135deg, #f0f4ff 0%, #e0e7ff 100%) !important;
        color: #1f2937 !important;
    }
    
    /* Spinner button container */
    .stNumberInput [data-baseweb="spinner"] {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%) !important;
    }
    
    .stNumberInput [data-baseweb="spinner"] button {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%) !important;
        color: #1f2937 !important;
        border-left: 1px solid #e2e8f0 !important;
    }
    
    .stNumberInput [data-baseweb="spinner"] button:hover {
        background: linear-gradient(135deg, #f0f4ff 0%, #e0e7ff 100%) !important;
        color: #1f2937 !important;
    }
    
    /* Spinner icons */
    .stNumberInput [data-baseweb="spinner"] svg {
        fill: #1f2937 !important;
        color: #1f2937 !important;
    }
    
    .stNumberInput > div > div:hover {
        border-color: #667eea !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.15) !important;
    }
    
    /* Enhanced checkbox styling */
    .stCheckbox {
        margin-bottom: 1rem !important;
    }
    
    .stCheckbox > label {
        font-family: 'Inter', sans-serif !important;
        font-size: 1rem !important;
        font-weight: 500 !important;
        color: #1f2937 !important;
        background: transparent !important;
        border: none !important;
        padding: 0.6rem 0 !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
    }
    
    .stCheckbox > label > div {
        color: #1f2937 !important;
    }
    
    .stCheckbox span {
        color: #1f2937 !important;
    }
    
    .stCheckbox input[type="checkbox"] {
        width: 20px !important;
        height: 20px !important;
        border: 2px solid #9ca3af !important;
        border-radius: 6px !important;
        background-color: #ffffff !important;
        margin-right: 0.8rem !important;
        transition: all 0.3s ease !important;
        appearance: none !important;
        -webkit-appearance: none !important;
    }
    
    .stCheckbox input[type="checkbox"]:checked {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border-color: #5a67d8 !important;
    }
    
    .stCheckbox input[type="checkbox"]:checked::after {
        content: "✓" !important;
        color: white !important;
        font-size: 14px !important;
        font-weight: bold !important;
        position: absolute !important;
        top: 50% !important;
        left: 50% !important;
        transform: translate(-50%, -50%) !important;
    }
    
    /* Enhanced metrics styling */
    .stMetric {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(0, 0, 0, 0.05);
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .stMetric [data-testid="metric-container"] {
        background: transparent;
        border: none;
        box-shadow: none;
    }
    
    .stMetric label {
        font-family: 'Inter', sans-serif !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        color: #64748b !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    
    .stMetric [data-testid="metric-container"] > div {
        font-family: 'Poppins', sans-serif !important;
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        color: #2c3e50 !important;
    }
    
    /* Skill category styling */
    .skill-category {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.8rem 0;
        border-left: 4px solid #667eea;
        box-shadow: 0 3px 12px rgba(0, 0, 0, 0.05);
    }
    
    .skill-category h4 {
        font-family: 'Poppins', sans-serif !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        color: #1f2937 !important;
        margin-bottom: 0.8rem !important;
        border-bottom: 1px solid #e2e8f0 !important;
        padding-bottom: 0.5rem !important;
    }
    
    .skill-category {
        color: #1f2937 !important;
    }
    
    .skill-category * {
        color: #1f2937 !important;
    }
    
    /* Info boxes */
    .info-box {
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        font-family: 'Inter', sans-serif;
    }
    
    .info-success {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border-left-color: #10b981;
        color: #064e3b;
    }
    
    .info-warning {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-left-color: #f59e0b;
        color: #78350f;
    }
    
    .info-primary {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        border-left-color: #3b82f6;
        color: #1e3a8a;
    }
    
    /* Enhanced spacing for main sections */
    .main-section {
        margin: 4rem 0;
        padding: 2rem 0;
    }
    
    /* Enhanced footer */
    .footer-section {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border-radius: 16px;
        padding: 1.5rem 1rem;
        margin-top: 2rem;
        text-align: center;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(0, 0, 0, 0.05);
    }
    
    .footer-section p {
        font-family: 'Inter', sans-serif !important;
        color: #64748b !important;
        line-height: 1.6 !important;
        margin: 0.8rem 0 !important;
    }
    
    .footer-section strong {
        color: #2c3e50 !important;
        font-weight: 600 !important;
    }
    
    /* Responsive design improvements */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.2rem;
        }
        
        .main .block-container {
            padding: 1rem 0.5rem;
            margin: 0.5rem;
        }
        
        .prediction-box {
            padding: 1rem;
        }
        
        .chart-container {
            padding: 1rem;
        }
    }
    
    /* Compact spacing utilities */
    .spacing-lg {
        margin: 1rem 0;
    }
    
    .spacing-xl {
        margin: 1.5rem 0;
    }
    
    /* Reduce default margins */
    .element-container {
        margin-bottom: 0.2rem !important;
    }
    
    /* Remove gaps between Streamlit elements */
    .stMarkdown {
        margin-bottom: 0 !important;
    }
    
    .stMarkdown > div {
        margin-bottom: 0 !important;
    }
    
    /* Remove empty space from containers */
    [data-testid="stVerticalBlock"] {
        gap: 0.2rem !important;
    }
    
    [data-testid="stHorizontalBlock"] {
        gap: 0.5rem !important;
    }
    
    /* Force all text to be dark - override any white text */
    .stApp, .stApp * {
        color: #1f2937 !important;
    }
    
    /* Specific overrides for common white text issues */
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] span,
    [data-testid="stMarkdownContainer"] div {
        color: #1f2937 !important;
    }
    
    .stCheckbox > label > div,
    .stCheckbox > label > div > span {
        color: #1f2937 !important;
    }
    
    /* Tab content text */
    .stTabs [data-baseweb="tab-panel"] {
        color: #1f2937 !important;
    }
    
    .stTabs [data-baseweb="tab-panel"] * {
        color: #1f2937 !important;
    }
    
    /* Skill selection text */
    .skill-category label,
    .skill-category span,
    .skill-category div {
        color: #1f2937 !important;
    }
    
    /* Enhanced scrollbar */
    ::-webkit-scrollbar {
        width: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 6px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #cbd5e1 0%, #94a3b8 100%);
        border-radius: 6px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #94a3b8 0%, #64748b 100%);
    }
    
    /* Additional fixes for white text issues */
    .stCheckbox > label[data-baseweb="checkbox"] {
        color: #1f2937 !important;
    }
    
    .stCheckbox > label[data-baseweb="checkbox"] > div {
        color: #1f2937 !important;
    }
    
    .stCheckbox > label[data-baseweb="checkbox"] > div > div {
        color: #1f2937 !important;
    }
    
    .stCheckbox input + div {
        color: #1f2937 !important;
    }
    
    /* Fix any remaining checkbox text */
    input[type="checkbox"] + div {
        color: #1f2937 !important;
    }
    
    /* Ensure visibility of all text elements */
    .stTextInput, .stSelectbox, .stNumberInput, .stSlider, .stRadio {
        color: #1f2937 !important;
    }
    
    .stTextInput label, .stSelectbox label, .stNumberInput label, 
    .stSlider label, .stRadio label {
        color: #1f2937 !important;
    }
    
    /* Make sure markdown text is visible */
    .stMarkdown {
        color: #1f2937 !important;
    }
    
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, 
    .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color: #1f2937 !important;
    }
    
    /* Tab panel content */
    [role="tabpanel"] {
        color: #1f2937 !important;
    }
    
    [role="tabpanel"] * {
        color: #1f2937 !important;
    }
    
    /* Override any inherited white colors */
    * {
        color: inherit !important;
    }
    
    body, .stApp {
        color: #1f2937 !important;
    }
    

    
    /* Simple menu styling */
    [data-baseweb="menu"] {
        background: #ffffff !important;
        border: 1px solid #d1d5db !important;
        border-radius: 6px !important;
    }
    
    [data-baseweb="menu"] li {
        background: #ffffff !important;
        color: #1f2937 !important;
        padding: 0.5rem 1rem !important;
    }
    
    [data-baseweb="menu"] li:hover {
        background: #f3f4f6 !important;
        color: #1f2937 !important;
    }
    
    /* Ensure selectbox dropdown arrow is visible */
    .stSelectbox [data-baseweb="select"] [title="open"] {
        color: #1f2937 !important;
    }
    
    /* Fix selectbox value display */
    .stSelectbox [data-baseweb="select"] span {
        color: #1f2937 !important;
        font-weight: 500 !important;
    }
    
    /* Selectbox placeholder and value text */
    .stSelectbox [data-baseweb="select"] div[data-baseweb="input"] {
        color: #1f2937 !important;
    }
    
    .stSelectbox [data-baseweb="select"] div[data-baseweb="input"] > div {
        color: #1f2937 !important;
    }
    
    /* All selectbox internal text */
    .stSelectbox [data-baseweb="select"] * {
        color: #1f2937 !important;
    }
    
    /* Selectbox root element text */
    .stSelectbox div[data-baseweb="select"] {
        color: #1f2937 !important;
    }
    
    /* Universal dropdown menu fixes */
    div[data-baseweb="popover"] {
        background-color: #ffffff !important;
    }
    
    div[data-baseweb="popover"] ul {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
    }
    
    div[data-baseweb="popover"] li {
        background-color: #ffffff !important;
        color: #1f2937 !important;
    }
    
    /* Streamlit selectbox dropdown override */
    .stSelectbox div[role="listbox"] {
        background-color: #ffffff !important;
    }
    
    .stSelectbox div[role="listbox"] div {
        background-color: #ffffff !important;
        color: #1f2937 !important;
    }
    

    
    /* Remove empty spaces in columns */
    .stColumn {
        padding: 0 !important;
    }
    
    .stColumn > div {
        padding: 0.2rem !important;
    }
    
    /* Remove chart container margins */
    .js-plotly-plot {
        margin: 0 !important;
    }
    
    /* Remove Streamlit default gaps */
    .block-container > div {
        gap: 0.3rem !important;
    }
    
    /* Compact metric spacing */
    [data-testid="metric-container"] {
        margin: 0.2rem 0 !important;
        padding: 0.5rem !important;
    }
    
    /* Remove header extra spacing */
    .stHeader {
        padding: 0 !important;
        margin: 0.5rem 0 !important;
    }
    
    /* Hide empty chart containers */
    .chart-container:empty {
        display: none !important;
    }
    
    /* Hide chart containers with only whitespace */
    .chart-container:blank {
        display: none !important;
    }
    
    /* Hide chart containers until content loads */
    .chart-container:not(:has(*)) {
        display: none !important;
    }
    
    /* Alternative: Hide containers with minimal content */
    .chart-container {
        min-height: 50px;
    }
    
    .chart-container:empty,
    .chart-container:not(:has(div)) {
        display: none !important;
        height: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_info():
    """Load the trained model and metadata"""
    try:
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
    """Calculate skill family scores based on selected skills"""
    scores = {}
    binary_flags = {}
    
    for family_name, family_techs in SKILL_FAMILIES.items():
        selected_from_family = [skill for skill in selected_skills if skill in family_techs]
        selected_count = len(selected_from_family)
        total_count = len(family_techs)
        
        percentage = (selected_count / total_count) * 100 if total_count > 0 else 0
        
        scores[f'{family_name}_Score'] = round(percentage, 1)
        binary_flags[f'Has_{family_name}'] = 1 if selected_count > 0 else 0
    
    return scores, binary_flags

def calculate_derived_features(binary_flags):
    """Calculate skill breadth and full-stack indicator"""
    skill_breadth = sum(binary_flags.values())
    
    is_fullstack = 1 if (binary_flags['Has_Programming'] == 1 and 
                        binary_flags['Has_Web'] == 1 and 
                        binary_flags['Has_Database'] == 1) else 0
    
    return {
        'Skill_Breadth': skill_breadth,
        'Is_FullStack': is_fullstack
    }

def create_feature_vector(user_inputs, feature_names):
    """Create feature vector from user inputs"""
    features = pd.DataFrame(0, index=[0], columns=feature_names)
    
    scores, binary_flags = calculate_skill_scores(user_inputs['selected_skills'])
    derived_features = calculate_derived_features(binary_flags)
    
    for score_name, score_value in scores.items():
        if score_name in feature_names:
            features.loc[0, score_name] = score_value
    
    for flag_name, flag_value in binary_flags.items():
        if flag_name in feature_names:
            features.loc[0, flag_name] = flag_value
    
    for derived_name, derived_value in derived_features.items():
        if derived_name in feature_names:
            features.loc[0, derived_name] = derived_value
    
    if 'ComputerSkills' in feature_names:
        skill_count = len(user_inputs['selected_skills'])
        features.loc[0, 'ComputerSkills'] = min(skill_count, 40)
    
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
    """Apply reality check to predictions"""
    skill_count = len(user_inputs['selected_skills'])
    
    adjustment_factors = []
    
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
    
    if derived['Is_FullStack']:
        adjustment_factors.append(("Full-stack capability", +0.1))
    
    if skill_count >= 15:
        adjustment_factors.append(("Extensive skills", +0.1))
    
    total_adjustment = sum(factor[1] for factor in adjustment_factors)
    adjusted_probability = max(0.05, min(0.95, probability + total_adjustment))
    
    return adjusted_probability, adjustment_factors

def get_recommendations(probability, user_inputs):
    """Generate personalized recommendations"""
    recommendations = []
    
    scores, binary_flags = calculate_skill_scores(user_inputs['selected_skills'])
    skill_count = len(user_inputs['selected_skills'])
    
    if skill_count <= 3:
        recommendations.append("**CRITICAL:** Learn at least 5-8 core technologies")
        recommendations.append("Focus on popular languages: Python, JavaScript, or Java")
    
    if not user_inputs['prof_experience'] and not user_inputs['is_developer']:
        recommendations.append("**URGENT:** Gain practical experience through projects or internships")
    
    if scores['Programming_Score'] < 30:
        recommendations.append("Master at least 2-3 programming languages")
    if scores['Web_Score'] < 20 and scores['Programming_Score'] > 30:
        recommendations.append("Add web development skills (HTML/CSS, React, Node.js)")
    if scores['Database_Score'] < 15:
        recommendations.append("Learn database fundamentals (SQL, MongoDB)")
    if scores['CloudDevOps_Score'] < 15 and skill_count > 8:
        recommendations.append("Add cloud/DevOps skills for senior roles")
    
    if probability < 0.4:
        recommendations.append("Focus on building a strong foundation before applying")
        recommendations.append("Consider bootcamps or formal education")
    elif probability < 0.6:
        recommendations.append("You're on the right track - keep learning!")
        recommendations.append("Network with developers in your areas of interest")
    elif probability < 0.8:
        recommendations.append("Start applying to positions matching your skills")
        recommendations.append("Highlight your technical strengths in applications")
    else:
        recommendations.append("You have strong employability - apply confidently!")
        recommendations.append("Consider specialized or senior positions")
    
    return recommendations[:5]

def main():
    # Header
    st.markdown('<h1 class="main-header">PredSeeker</h1>', unsafe_allow_html=True)
    st.markdown('<h3 class="sub-header">AI-Powered Developer Employment Predictor</h3>', unsafe_allow_html=True)
    
    # Load model
    model, model_info = load_model_and_info()
    
    if model is None:
        st.error("Could not load the trained model. Please check the model files.")
        st.info("Make sure you have run the model training notebook and the model files exist in the 'models' directory.")
        return
    
    # Enhanced Sidebar
    with st.sidebar:
        # Fixed light mode only
        appearance = "Light"
        if appearance == "Dark":
            # Apply a dark-mode class for consistent targeting
            st.markdown("""
            <script>
            const root = document.documentElement;
            root.classList.add('dark-mode');
            </script>
            """, unsafe_allow_html=True)
            st.markdown("""
            <script>
            const r = document.documentElement;
            r.setAttribute('data-theme','dark');
            </script>
            """, unsafe_allow_html=True)
            st.markdown(
                """
                <style>
                [data-theme='dark'] {
                    --bg: #0b1220;
                    --panel: #0f172a;
                    --panel-2: #111827;
                    --text: #e5e7eb;
                    --muted: #a1a1aa;
                    --accent1: #818cf8;
                    --accent2: #a78bfa;
                    --border: rgba(255,255,255,0.08);
                }

                /* Strong dark overrides so they win over earlier light styles */
                [data-theme='dark'] .stApp, [data-theme='dark'] .stApp *,
                [data-theme='dark'] [data-testid="stMarkdownContainer"] *,
                [data-theme='dark'] .stTabs [data-baseweb="tab-panel"] * { color: var(--text) !important; }

                [data-theme='dark'] .stApp { background: linear-gradient(135deg, var(--bg) 0%, var(--panel-2) 100%) !important; }
                [data-theme='dark'] .main .block-container { background: rgba(15, 23, 42, 0.92) !important; box-shadow: 0 12px 30px rgba(0,0,0,0.6) !important; }

                /* Headings */
                [data-theme='dark'] .main-header { background: linear-gradient(135deg, var(--accent1) 0%, var(--accent2) 100%) !important; -webkit-background-clip: text !important; -webkit-text-fill-color: transparent !important; }
                [data-theme='dark'] .prediction-result-header, [data-theme='dark'] .chart-title, [data-theme='dark'] h1, [data-theme='dark'] h2, [data-theme='dark'] h3 { color: var(--text) !important; }

                /* Panels */
                [data-theme='dark'] .prediction-container, [data-theme='dark'] .chart-container, [data-theme='dark'] .sidebar-section, [data-theme='dark'] .info-box, [data-theme='dark'] .footer-section { 
                    background: linear-gradient(135deg, var(--panel) 0%, var(--panel-2) 100%) !important; 
                    border-color: var(--border) !important;
                }
                [data-theme='dark'] .info-success { background: linear-gradient(135deg, rgba(16,185,129,0.15) 0%, rgba(16,185,129,0.10) 100%) !important; color: #d1fae5 !important; border-left-color: #34d399 !important; }
                [data-theme='dark'] .info-warning { background: linear-gradient(135deg, rgba(245,158,11,0.15) 0%, rgba(245,158,11,0.10) 100%) !important; color: #fde68a !important; border-left-color: #f59e0b !important; }
                [data-theme='dark'] .info-primary { background: linear-gradient(135deg, rgba(59,130,246,0.18) 0%, rgba(59,130,246,0.12) 100%) !important; color: #bfdbfe !important; border-left-color: #60a5fa !important; }

                /* Tabs */
                [data-theme='dark'] .stTabs [data-baseweb="tab-list"] { background: linear-gradient(135deg, #0b1220, #0f172a) !important; border: 1px solid var(--border) !important; box-shadow: none !important; }
                [data-theme='dark'] .stTabs [data-baseweb="tab"] { background: #0f172a !important; color: var(--text) !important; border: 1px solid transparent !important; }
                [data-theme='dark'] .stTabs [data-baseweb="tab"] * { color: var(--text) !important; }
                [data-theme='dark'] .stTabs [data-baseweb="tab"]:hover { border-color: var(--accent1) !important; background: #111827 !important; }
                [data-theme='dark'] .stTabs [aria-selected="true"] { background: linear-gradient(135deg, var(--accent1) 0%, var(--accent2) 100%) !important; color: #0b1020 !important; }

                /* Inputs */
                [data-theme='dark'] .stSelectbox > div > div, [data-theme='dark'] .stNumberInput > div > div { background: #0f172a !important; border: 2px solid #334155 !important; box-shadow: none !important; }
                [data-theme='dark'] .stSelectbox [data-baseweb="select"] { background: transparent !important; }
                [data-theme='dark'] .stSelectbox [role="listbox"] { background: #0f172a !important; border: 1px solid #334155 !important; }
                [data-theme='dark'] .stSelectbox [role="option"] { background: transparent !important; color: var(--text) !important; }
                [data-theme='dark'] .stSelectbox [role="option"]:hover { background: #111827 !important; }
                [data-theme='dark'] .stNumberInput input { background: transparent !important; color: var(--text) !important; }
                [data-theme='dark'] .stCheckbox input[type="checkbox"] { border-color: #64748b !important; background: #0f172a !important; }
                [data-theme='dark'] .stCheckbox > label, [data-theme='dark'] .stCheckbox span, [data-theme='dark'] .stCheckbox > label > div { color: var(--text) !important; }

                /* Sidebar */
                [data-theme='dark'] [data-testid="stSidebar"], [data-theme='dark'] .css-1d391kg { background: linear-gradient(180deg, var(--panel) 0%, var(--panel-2) 100%) !important; border-right: 2px solid var(--border) !important; }
                [data-theme='dark'] .sidebar-header { border-bottom-color: #334155 !important; }
                [data-theme='dark'] .sidebar-section div[style*='color: #374151'] { color: var(--text) !important; }

                /* Metrics */
                [data-theme='dark'] .stMetric { background: #0f172a !important; border-color: var(--border) !important; }
                [data-theme='dark'] .stMetric label { color: #cbd5e1 !important; }

                /* Buttons */
                [data-theme='dark'] .stButton > button { box-shadow: 0 8px 26px rgba(129, 140, 248, 0.35) !important; }
                [data-theme='dark'] .stButton > button:hover { filter: brightness(1.03); }

                /* Skill category card */
                [data-theme='dark'] .skill-category { background: #0f172a !important; border-left-color: var(--accent1) !important; box-shadow: 0 3px 12px rgba(0,0,0,0.35) !important; }

                /* Force all high-level containers to dark background */
                [data-theme='dark'] body, 
                [data-theme='dark'] .stApp,
                [data-theme='dark'] [data-testid='stAppViewContainer'],
                [data-theme='dark'] .main,
                [data-theme='dark'] .block-container { background: #0b1220 !important; }

                /* Remove light container around tabs */
                [data-theme='dark'] .stTabs { background: transparent !important; }
                [data-theme='dark'] .stTabs > div { background: transparent !important; }
                </style>
                """,
                unsafe_allow_html=True,
            )
            # Extra high-specificity hard overrides (covers widgets that inject inline styles)
            st.markdown(
                """
                <style>
                .stApp, .stApp * { color: #e5e7eb !important; }
                .main .block-container { background: #0f172a !important; }
                section[data-testid="stSidebar"], .css-1d391kg { background: #0f172a !important; }
                .stTabs [data-baseweb="tab-list"] { background: #0f172a !important; }
                .stTabs [data-baseweb="tab"] { background: #111827 !important; color: #e5e7eb !important; }
                .prediction-container, .chart-container, .sidebar-section, .info-box, .footer-section, .skill-category { background: #0f172a !important; border-color: rgba(255,255,255,0.08) !important; }
                .stSelectbox > div > div, .stNumberInput > div > div { background: #0f172a !important; border-color: #334155 !important; }
                .stSelectbox [role="listbox"] { background: #0f172a !important; border-color: #334155 !important; }
                .stSelectbox [role="option"] { color: #e5e7eb !important; }
                /* Fix inline light text within sidebar info blocks */
                .sidebar-section div[style*='color: #374151'] { color: #e5e7eb !important; }
                /* Force page background */
                html, body, .stApp, [data-testid="stAppViewContainer"] { background: #0b1220 !important; }
                .stTabs [data-baseweb="tab-panel"] { background: transparent !important; }
                /* Nuke any inline white backgrounds Streamlit injects */
                [data-theme='dark'] div[style*="background: rgb(255, 255, 255)"],
                [data-theme='dark'] div[style*="background-color: rgb(255, 255, 255)"],
                [data-theme='dark'] div[style*="background: #fff"],
                [data-theme='dark'] div[style*="background-color: #fff"],
                [data-theme='dark'] div[style*="background: white"],
                [data-theme='dark'] div[style*="background-color: white"] {
                    background: #0f172a !important;
                }
                /* All text becomes light in dark mode */
                [data-theme='dark'] * { color: #e5e7eb !important; }
                /* Borders and dividers */
                [data-theme='dark'] hr, [data-theme='dark'] .stDivider, [data-theme='dark'] .stMarkdown hr { border-color: #334155 !important; color: #334155 !important; }
                </style>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown("""
            <script>
            const r = document.documentElement; r.setAttribute('data-theme','light'); r.classList.remove('dark-mode');
            </script>
            """, unsafe_allow_html=True)
        st.markdown("""
        <div class="sidebar-section">
            <h3 class="sidebar-header">Model Performance</h3>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("ROC-AUC", f"{model_info['performance_metrics']['ROC-AUC']:.3f}")
            st.metric("Accuracy", f"{model_info['performance_metrics']['Accuracy']:.3f}")
        with col2:
            st.metric("F1-Score", f"{model_info['performance_metrics']['F1-Score']:.3f}")
            st.metric("Precision", f"{model_info['performance_metrics']['Precision']:.3f}")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="sidebar-section">
            <h3 class="sidebar-header">Model Information</h3>
            <div style="color: #374151; line-height: 1.8; font-size: 1rem;">
                <p><strong>Algorithm:</strong> {model_info['model_name']}</p>
                <p><strong>Features:</strong> {len(model_info['features'])}</p>
                <p><strong>Training Date:</strong> {model_info['training_date']}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="sidebar-section">
            <h3 class="sidebar-header">Dataset Statistics</h3>
            <div style="color: #374151; line-height: 1.8; font-size: 1rem;">
                <p><strong>Training Samples:</strong> 58,769</p>
                <p><strong>Test Samples:</strong> 14,693</p>
                <p><strong>Data Source:</strong> Stack Overflow Survey</p>
                <p><strong>Employment Rate:</strong> ~74% (Balanced Dataset)</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box info-primary">
            <p style="margin: 0; font-size: 0.95rem; text-align: center;">
                <strong>Pro Tip:</strong> Select diverse skills across multiple categories for better employment predictions!
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content with enhanced tabs
    tab1, tab2 = st.tabs(["Make Prediction", "About the Model"])
    
    with tab1:
        col1, col2 = st.columns([2, 1], gap="large")
        
        with col1:
            st.header("Enter Your Developer Profile")
            
            # Skills Selection Section
            st.subheader("Technical Skills - Select Your Skills")
            
            tab_prog, tab_web, tab_db, tab_cloud = st.tabs(["Programming", "Web Tech", "Database", "Cloud/DevOps"])
            
            selected_skills = []
            
            with tab_prog:
                st.markdown('<div class="skill-category">', unsafe_allow_html=True)
                st.markdown('<h4>Programming Languages & Frameworks</h4>', unsafe_allow_html=True)
                prog_cols = st.columns(3)
                for i, skill in enumerate(SKILL_FAMILIES['Programming']):
                    with prog_cols[i % 3]:
                        if st.checkbox(skill, key=f"prog_{skill}"):
                            selected_skills.append(skill)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with tab_web:
                st.markdown('<div class="skill-category">', unsafe_allow_html=True)
                st.markdown('<h4>Web Development Technologies</h4>', unsafe_allow_html=True)
                web_cols = st.columns(3)
                for i, skill in enumerate(SKILL_FAMILIES['Web']):
                    with web_cols[i % 3]:
                        if st.checkbox(skill, key=f"web_{skill}"):
                            selected_skills.append(skill)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with tab_db:
                st.markdown('<div class="skill-category">', unsafe_allow_html=True)
                st.markdown('<h4>Database & Data Storage</h4>', unsafe_allow_html=True)
                db_cols = st.columns(3)
                for i, skill in enumerate(SKILL_FAMILIES['Database']):
                    with db_cols[i % 3]:
                        if st.checkbox(skill, key=f"db_{skill}"):
                            selected_skills.append(skill)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with tab_cloud:
                st.markdown('<div class="skill-category">', unsafe_allow_html=True)
                st.markdown('<h4>Cloud Computing & DevOps</h4>', unsafe_allow_html=True)
                cloud_cols = st.columns(3)
                for i, skill in enumerate(SKILL_FAMILIES['CloudDevOps']):
                    with cloud_cols[i % 3]:
                        if st.checkbox(skill, key=f"cloud_{skill}"):
                            selected_skills.append(skill)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Display selected skills summary
            if selected_skills:
                st.markdown(f"""
                <div class="info-box info-success">
                    <strong>Selected {len(selected_skills)} skills:</strong><br>
                    <span>{', '.join(selected_skills[:5])}{' and ' + str(len(selected_skills) - 5) + ' more...' if len(selected_skills) > 5 else ''}</span>
                </div>
                """, unsafe_allow_html=True)
                
                scores, binary_flags = calculate_skill_scores(selected_skills)
                derived = calculate_derived_features(binary_flags)
                st.subheader("Your Skill Profile")
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
                
                if derived['Is_FullStack']:
                    st.markdown("""
                    <div class="info-box info-success">
                        <strong>Full-Stack Developer Profile Detected!</strong>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="info-box info-primary">
                    <strong>Skill Breadth: {derived['Skill_Breadth']} out of 4 skill categories</strong>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="info-box info-warning">
                    <strong>Please select at least one skill to continue</strong>
                </div>
                """, unsafe_allow_html=True)
            
            # Demographics Section
            st.subheader("Personal Information")
            
            st.markdown("#### Age")
            age = st.number_input(
                "Enter your age", 
                min_value=16, 
                max_value=70, 
                value=28,
                key="age_input",
                help="Your current age in years"
            )
            
            st.markdown("#### Gender")
            gender_options = ["Man", "Woman", "Prefer not to say"]
            gender_mapping = {"Man": "Man", "Woman": "Woman", "Prefer not to say": "NonBinary"}
            gender_display = st.selectbox(
                "Select your gender:",
                gender_options,
                index=0,
                key="gender_select"
            )
            gender = gender_mapping[gender_display]
            
            # Professional Section
            st.subheader("Professional Background")
            
            # Computer skills display
            total_selected_skills = len(selected_skills)
            
            if total_selected_skills > 0:
                prog_selected = len([s for s in selected_skills if s in SKILL_FAMILIES['Programming']])
                web_selected = len([s for s in selected_skills if s in SKILL_FAMILIES['Web']])
                db_selected = len([s for s in selected_skills if s in SKILL_FAMILIES['Database']])
                cloud_selected = len([s for s in selected_skills if s in SKILL_FAMILIES['CloudDevOps']])
                
                st.markdown(f"""
                <div class="info-box info-primary" style="text-align: center;">
                    <h3 style="margin-bottom: 1rem;">Computer Skills Count: {total_selected_skills}</h3>
                    <p style="margin: 0; font-size: 1.1rem;">
                        <strong>Auto-calculated Breakdown:</strong><br>
                        Programming({prog_selected}) + Web({web_selected}) + Database({db_selected}) + Cloud/DevOps({cloud_selected}) = {total_selected_skills} total skills
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="info-box info-primary" style="text-align: center;">
                    <h3 style="margin-bottom: 0.5rem;">Computer Skills Count: 0</h3>
                    <p style="margin: 0;">Select skills above to update automatically</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Education Level
            st.markdown("#### Education Level")
            
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
                index=2,
                key="education_select"
            )
            education_level = education_mapping[education_display]
            
            # Experience and Background
            st.subheader("Experience & Background")
            col1_exp, col2_exp = st.columns(2)
            with col1_exp:
                is_developer = st.checkbox("I am a professional developer", value=True)
                prof_experience = st.checkbox("I have professional coding experience", value=True)
            with col2_exp:
                mental_health = st.checkbox("I have mental health concerns")
                accessibility = st.checkbox("I have accessibility needs")
            
            salary_info = st.checkbox("I'm willing to share salary information")
            
            # Prediction Button
            if not selected_skills:
                st.markdown("""
                <div class="info-box info-warning" style="text-align: center;">
                    <strong>Please select at least one technical skill above to enable prediction</strong>
                </div>
                """, unsafe_allow_html=True)
                st.button("Predict Employment Probability", disabled=True, use_container_width=True)
            elif st.button("Predict Employment Probability", type="primary", use_container_width=True):
                user_inputs = {
                    'selected_skills': selected_skills,
                    'age': age,
                    'gender': gender,
                    'computer_skills': len(selected_skills),
                    'education_level': education_level,
                    'is_developer': is_developer,
                    'prof_experience': prof_experience,
                    'mental_health': mental_health,
                    'accessibility': accessibility,
                    'salary_info': salary_info
                }
                
                try:
                    features = create_feature_vector(user_inputs, model_info['features'])
                    
                    skill_count = len(user_inputs['selected_skills'])
                    scores, binary_flags = calculate_skill_scores(user_inputs['selected_skills'])
                    derived = calculate_derived_features(binary_flags)
                    
                    if skill_count < 3 and not user_inputs['prof_experience']:
                        st.warning("Note: Limited skills and no professional experience may impact employability.")
                    
                    raw_prediction = model.predict(features)[0]
                    raw_probability = model.predict_proba(features)[0][1]
                    
                    adjusted_probability, _ = reality_check_prediction(
                        raw_probability, user_inputs, scores, derived
                    )
                    
                    prediction = 1 if adjusted_probability >= 0.5 else 0
                    probability = adjusted_probability
                    
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
        
        # Enhanced Results Display
        with col2:
            if 'prediction' in st.session_state:
                prediction = st.session_state['prediction']
                probability = st.session_state['probability']
                user_inputs = st.session_state['user_inputs']
                
                st.markdown('<div class="prediction-container">', unsafe_allow_html=True)
                st.markdown('<h2 class="prediction-result-header">Prediction Result</h2>', unsafe_allow_html=True)
                
                if prediction == 1:
                    st.markdown(f"""
                    <div class="prediction-box employed">
                        <h2>High Employment Probability</h2>
                        <h3>Success Rate: {probability:.1%}</h3>
                        <p>Your technical profile shows strong employment potential!</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-box unemployed">
                        <h2>Room for Improvement</h2>
                        <h3>Current Probability: {probability:.1%}</h3>
                        <p>Consider expanding your technical skillset for better opportunities.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Progress visualization
                st.progress(int(probability * 100), text=f"Employment probability: {probability:.1%}")

                # Celebration for high scores
                if probability >= 0.8:
                    st.balloons()

                # Only show gauge if prediction exists
                if 'prediction' in st.session_state:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    st.markdown('<h4 class="chart-title">Employment Probability Gauge</h4>', unsafe_allow_html=True)
                    
                    # Your gauge chart code here...
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = probability * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        number = {'font': {'size': 32, 'color': '#2c3e50', 'family': 'Poppins'}},
                        gauge = {
                            'axis': {
                                'range': [None, 100], 
                                'tickcolor': '#64748b',
                                'tickfont': {'size': 14, 'color': '#64748b'}
                            },
                            'bar': {'color': "#667eea", 'thickness': 0.7},
                            'steps': [
                                {'range': [0, 40], 'color': "#fee2e2"},
                                {'range': [40, 70], 'color': "#fef3c7"},
                                {'range': [70, 100], 'color': "#d1fae5"}
                            ],
                            'threshold': {
                                'line': {'color': "#ef4444", 'width': 4},
                                'thickness': 0.8,
                                'value': 50
                            },
                            'bordercolor': "#e2e8f0",
                            'borderwidth': 2
                        }
                    ))
                    fig.update_layout(
                        height=280, 
                        margin=dict(l=10, r=10, t=10, b=10),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font={'family': 'Inter, sans-serif', 'size': 14}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Only show radar chart if prediction exists  
                if 'prediction' in st.session_state:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    st.markdown('<h4 class="chart-title">Skills Profile Analysis</h4>', unsafe_allow_html=True)
                    
                    # Your radar chart code here...
                    fig_radar = go.Figure()
                    scores, _ = calculate_skill_scores(user_inputs['selected_skills'])
                    skills_data = {
                        'Programming': scores['Programming_Score'],
                        'Web Tech': scores['Web_Score'],
                        'Database': scores['Database_Score'],
                        'Cloud/DevOps': scores['CloudDevOps_Score']
                    }
                    
                    fig_radar.add_trace(go.Scatterpolar(
                        r=list(skills_data.values()),
                        theta=list(skills_data.keys()),
                        fill='toself',
                        name='Your Skills',
                        line=dict(color='#667eea', width=4),
                        fillcolor='rgba(102, 126, 234, 0.25)',
                        marker=dict(color='#667eea', size=10, symbol='circle')
                    ))
                    fig_radar.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 100],
                                tickfont=dict(size=12, color='#64748b', family='Inter'),
                                gridcolor='#e2e8f0',
                                linecolor='#cbd5e1'
                            ),
                            angularaxis=dict(
                                tickfont=dict(size=14, color='#374151', family='Inter', weight=500),
                                gridcolor='#e2e8f0',
                                linecolor='#cbd5e1'
                            ),
                            bgcolor='rgba(0,0,0,0)'
                        ),
                        showlegend=False,
                        height=280,
                        margin=dict(l=10, r=10, t=10, b=10),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font={'family': 'Inter, sans-serif'}
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Enhanced recommendations
                st.subheader("Personalized Recommendations")
                recommendations = get_recommendations(probability, user_inputs)
                for i, rec in enumerate(recommendations, 1):
                    st.markdown(f"**{i}.** {rec}")
                
                # Enhanced PDF report with attractive styling
                from io import BytesIO
                from reportlab.lib.pagesizes import A4
                from reportlab.pdfgen import canvas
                from reportlab.lib.colors import HexColor, black, white
                from reportlab.lib.units import inch
                from reportlab.lib import colors
                
                buffer = BytesIO()
                c = canvas.Canvas(buffer, pagesize=A4)
                width, height = A4
                
                # Color scheme
                primary_color = HexColor('#667eea')
                secondary_color = HexColor('#764ba2')
                success_color = HexColor('#10b981')
                warning_color = HexColor('#f59e0b')
                text_color = HexColor('#1f2937')
                light_gray = HexColor('#f8fafc')
                
                # Header with gradient effect
                c.setFillColor(primary_color)
                c.rect(0, height - 80, width, 80, fill=True, stroke=False)
                
                # Title
                c.setFillColor(white)
                c.setFont("Helvetica-Bold", 24)
                c.drawCentredString(width/2, height - 35, "PredSeeker")
                c.setFont("Helvetica", 14)
                c.drawCentredString(width/2, height - 55, "AI-Powered Developer Employment Prediction Report")
                
                y = height - 120
                
                # Report info box
                c.setFillColor(light_gray)
                c.rect(40, y - 60, width - 80, 60, fill=True, stroke=True)
                c.setFillColor(text_color)
                c.setFont("Helvetica", 10)
                c.drawString(50, y - 20, f"Generated: {datetime.utcnow().strftime('%B %d, %Y at %I:%M %p UTC')}")
                c.drawString(50, y - 35, f"Report ID: PRED-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}")
                c.drawString(50, y - 50, "Data Source: Stack Overflow Developer Survey")
                
                y -= 100
                
                # Prediction result with colored background
                status = "EMPLOYED" if prediction == 1 else "NEEDS IMPROVEMENT"
                status_color = success_color if prediction == 1 else warning_color
                
                c.setFillColor(status_color)
                c.rect(40, y - 50, width - 80, 50, fill=True, stroke=False)
                
                c.setFillColor(white)
                c.setFont("Helvetica-Bold", 20)
                c.drawCentredString(width/2, y - 20, f"PREDICTION: {status}")
                
                c.setFont("Helvetica-Bold", 16)
                prob_text = f"Employment Probability: {probability:.1%}"
                c.drawCentredString(width/2, y - 40, prob_text)
                
                y -= 80
                
                # Skills analysis section
                c.setFillColor(text_color)
                c.setFont("Helvetica-Bold", 16)
                c.drawString(40, y, "SKILLS ANALYSIS")
                y -= 25
                
                scores, binary_flags = calculate_skill_scores(user_inputs['selected_skills'])
                derived = calculate_derived_features(binary_flags)
                
                # Skills breakdown
                skill_categories = [
                    ("Programming Languages", scores['Programming_Score'], len([s for s in user_inputs['selected_skills'] if s in SKILL_FAMILIES['Programming']])),
                    ("Web Technologies", scores['Web_Score'], len([s for s in user_inputs['selected_skills'] if s in SKILL_FAMILIES['Web']])),
                    ("Database Systems", scores['Database_Score'], len([s for s in user_inputs['selected_skills'] if s in SKILL_FAMILIES['Database']])),
                    ("Cloud & DevOps", scores['CloudDevOps_Score'], len([s for s in user_inputs['selected_skills'] if s in SKILL_FAMILIES['CloudDevOps']]))
                ]
                
                for category, score, count in skill_categories:
                    if y < 100:
                        c.showPage()
                        y = height - 50
                    
                    c.setFont("Helvetica-Bold", 12)
                    c.drawString(50, y, f"{category}:")
                    c.setFont("Helvetica", 11)
                    c.drawString(50, y - 15, f"  • Score: {score:.1f}%")
                    c.drawString(50, y - 30, f"  • Skills Count: {count}")
                    
                    # Progress bar
                    bar_width = 200
                    bar_height = 8
                    c.setFillColor(HexColor('#e2e8f0'))
                    c.rect(50, y - 45, bar_width, bar_height, fill=True, stroke=False)
                    c.setFillColor(primary_color)
                    c.rect(50, y - 45, bar_width * (score/100), bar_height, fill=True, stroke=False)
                    
                    y -= 70
                
                # Additional metrics
                if y < 120:
                    c.showPage()
                    y = height - 50
                
                c.setFont("Helvetica-Bold", 16)
                c.drawString(40, y, "ADDITIONAL METRICS")
                y -= 25
                
                metrics = [
                    ("Total Skills", len(user_inputs['selected_skills'])),
                    ("Skill Breadth", f"{derived['Skill_Breadth']}/4 categories"),
                    ("Full-Stack Developer", "Yes" if derived['Is_FullStack'] else "No"),
                    ("Professional Experience", "Yes" if user_inputs['prof_experience'] else "No"),
                    ("Developer Role", "Yes" if user_inputs['is_developer'] else "No")
                ]
                
                for metric, value in metrics:
                    if y < 60:
                        c.showPage()
                        y = height - 50
                    c.setFont("Helvetica-Bold", 11)
                    c.drawString(50, y, f"{metric}:")
                    c.setFont("Helvetica", 11)
                    c.drawString(50, y - 15, str(value))
                    y -= 30
                
                # Recommendations section
                if y < 150:
                    c.showPage()
                    y = height - 50
                
                c.setFont("Helvetica-Bold", 16)
                c.drawString(40, y, "PERSONALIZED RECOMMENDATIONS")
                y -= 25
                
                recommendations = get_recommendations(probability, user_inputs)
                for i, rec in enumerate(recommendations, 1):
                    if y < 60:
                        c.showPage()
                        y = height - 50
                        c.setFont("Helvetica", 11)
                    
                    c.setFont("Helvetica-Bold", 11)
                    c.drawString(50, y, f"{i}.")
                    c.setFont("Helvetica", 11)
                    # Wrap long text
                    text = rec.replace('**', '').replace('*', '')
                    if len(text) > 80:
                        words = text.split()
                        lines = []
                        current_line = ""
                        for word in words:
                            if len(current_line + word) < 80:
                                current_line += word + " "
                            else:
                                lines.append(current_line.strip())
                                current_line = word + " "
                        lines.append(current_line.strip())
                        for line in lines:
                            c.drawString(70, y, line)
                            y -= 15
                    else:
                        c.drawString(70, y, text)
                    y -= 20
                
                # Footer
                c.showPage()
                y = height - 50
                c.setFillColor(primary_color)
                c.rect(0, 0, width, 50, fill=True, stroke=False)
                c.setFillColor(white)
                c.setFont("Helvetica", 10)
                c.drawCentredString(width/2, 20, "PredSeeker - AI-Powered Developer Employment Prediction")
                c.drawCentredString(width/2, 8, "Disclaimer: This prediction is for guidance only. Individual results may vary.")
                
                c.save()
                pdf_bytes = buffer.getvalue()
                buffer.close()
                st.download_button(
                    label="Download Report (PDF)",
                    data=pdf_bytes,
                    file_name="predseeker_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

                st.markdown('</div>', unsafe_allow_html=True)

    
    with tab2:
        st.header("About the Employment Prediction Model")
        
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.subheader("How It Works")
            st.markdown("""
            This AI model predicts developer employment probability using:
            
            **Machine Learning Algorithm:** XGBoost Gradient Boosting  
            **Training Data:** 73,462 developers from Stack Overflow Survey  
            **Features:** 21 engineered features including skills, demographics, and experience  
            **Accuracy:** 89.5% ROC-AUC score on test data  
            
            The model analyzes your technical skills, professional background, and demographic information to provide employment insights.
            """)
            
            st.subheader("Key Features")
            st.markdown("""
            **Skill Scores:** Programming, Web, Database, Cloud/DevOps  
            **Derived Metrics:** Skill breadth, full-stack capability  
            **Demographics:** Age, gender, accessibility needs  
            **Professional:** Experience, education, developer status  
            """)
        
        with col2:
            st.subheader("Model Performance")
            
            # Enhanced performance metrics chart
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown('<h4 class="chart-title">Performance Metrics</h4>', unsafe_allow_html=True)
            
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
            
            fig_metrics = go.Figure(data=[
                go.Bar(
                    x=metrics_data['Score'],
                    y=metrics_data['Metric'],
                    orientation='h',
                    marker=dict(
                        color=['#667eea', '#764ba2', '#667eea', '#764ba2', '#667eea'],
                        line=dict(color='rgba(0,0,0,0.1)', width=1)
                    ),
                    text=[f"{score:.3f}" for score in metrics_data['Score']],
                    textposition='outside',
                    textfont=dict(size=14, color='#2c3e50', family='Inter', weight=600)
                )
            ])
            
            fig_metrics.update_layout(
                height=320,
                margin=dict(l=10, r=40, t=10, b=10),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'family': 'Inter, sans-serif'},
                xaxis=dict(
                    gridcolor='#e2e8f0',
                    tickfont=dict(color='#64748b', size=12),
                    showgrid=True,
                    zeroline=False
                ),
                yaxis=dict(
                    gridcolor='#e2e8f0',
                    tickfont=dict(color='#374151', size=13, weight=500),
                    showgrid=False
                ),
                showlegend=False
            )
            st.plotly_chart(fig_metrics, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.subheader("Limitations")
            st.markdown("""
            Predictions are probabilistic, not guarantees  
            Based on survey data from specific time period  
            Individual circumstances may vary  
            Use as guidance, not definitive career advice  
            """)

    # Enhanced Footer
    st.markdown("""
    <div class="footer-section">
        <p style="font-size: 1.4rem; margin-bottom: 1.5rem;"><strong>PredSeeker</strong> - AI-Powered Developer Employment Prediction</p>
        <p style="margin-bottom: 1rem;">Built with Streamlit • Powered by XGBoost ML Algorithm</p>
        <p style="font-size: 0.95rem; color: #64748b; margin-bottom: 0;"><em>Disclaimer: This tool provides probabilistic predictions for guidance only. Individual results may vary.</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
