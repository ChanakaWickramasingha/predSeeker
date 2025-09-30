# PredSeeker Project - Technology Stack & Implementation Guide

## 🛠️ Complete Technology Stack

### **Backend Technologies:**
- **Python 3.13** - Core programming language
- **XGBoost 1.7+** - Gradient boosting ML algorithm
- **Scikit-learn 1.3+** - ML pipeline and preprocessing
- **Pandas 2.0+** - Data manipulation and analysis
- **NumPy 1.24+** - Numerical computing
- **Joblib 1.3+** - Model serialization and loading

### **Frontend Technologies:**
- **Streamlit 1.28+** - Web application framework
- **Plotly 5.15+** - Interactive data visualization
- **HTML5/CSS3** - Custom styling and layout
- **JavaScript (embedded)** - Interactive chart functionality

### **Development Tools:**
- **Jupyter Notebook** - Data analysis and model training
- **VS Code** - Primary development environment
- **Git** - Version control system
- **PowerShell** - Command line interface (Windows)

### **Data & Model Files:**
- **CSV Files** - Processed training and test datasets
- **Joblib Model** - Serialized XGBoost classifier
- **JSON Metadata** - Model performance and feature information
- **Markdown Documentation** - Project summaries and guides

---

## 📁 Project Structure

```
predSeeker/
├── streamlit_app.py           # Main web application (1,600+ lines)
├── requirements_streamlit.txt  # Streamlit dependencies
├── requirements.txt           # Core ML dependencies
├── README.md                 # Project overview
│
├── data/
│   ├── processed/
│   │   ├── preprocessed_data_clean.csv
│   │   ├── X_train.csv
│   │   ├── X_test.csv  
│   │   ├── y_train.csv
│   │   └── y_test.csv
│   └── raw/
│       └── stackoverflow_with_nulls.csv
│
├── models/
│   ├── best_employment_model.joblib    # Trained XGBoost model
│   ├── model_comparison.csv           # Model evaluation results
│   └── model_info.json              # Model metadata & performance
│
├── notebooks/
│   ├── 01_EDA.ipynb                  # Exploratory data analysis
│   └── 02_Model_Training.ipynb       # ML model development
│
├── src/
│   ├── eda.py                        # EDA functions
│   ├── feature_engineering.py        # Feature creation logic
│   ├── models.py                     # ML model training
│   ├── preprocessing_clean.py        # Data cleaning
│   └── preprocessing.py              # Data preprocessing
│
└── results/
    ├── FINAL_PROJECT_SUMMARY.md      # Complete project overview
    ├── STREAMLIT_APP_SUMMARY.md      # Web application details  
    ├── TECH_STACK_GUIDE.md          # This file
    ├── preprocessing_eda_summary.md   # Data analysis results
    ├── preprocessing_final_summary.md # Final preprocessing report
    └── TEAM_PREPROCESSING_SUMMARY.md # Team collaboration summary
```

---

## ⚙️ Installation & Setup Guide

### **Prerequisites:**
- Python 3.9+ (Recommended: Python 3.13)
- Git for version control
- 4GB+ RAM for model training
- Modern web browser for application access

### **1. Environment Setup:**
```bash
# Clone the repository
git clone https://github.com/ChanakaWickramasingha/predSeeker.git
cd predSeeker

# Create virtual environment (recommended)
python -m venv predseeker_env

# Activate virtual environment
# Windows:
predseeker_env\Scripts\activate
# macOS/Linux:
source predseeker_env/bin/activate
```

### **2. Install Dependencies:**
```bash
# Install core ML dependencies
pip install -r requirements.txt

# Install Streamlit-specific dependencies  
pip install -r requirements_streamlit.txt

# Verify installation
pip list | grep -E "(streamlit|xgboost|pandas|plotly)"
```

### **3. Model Training (Optional):**
```bash
# Run EDA notebook
jupyter notebook notebooks/01_EDA.ipynb

# Run model training notebook  
jupyter notebook notebooks/02_Model_Training.ipynb

# Models will be saved to models/ directory
```

### **4. Launch Web Application:**
```bash
# Start Streamlit application
streamlit run streamlit_app.py --server.port 8502

# Access application
# Browser will open automatically or visit:
# http://localhost:8502
```

---

## 📦 Dependency Management

### **Core Requirements (requirements.txt):**
```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=1.7.0
joblib>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
jupyter>=1.0.0
```

### **Streamlit Requirements (requirements_streamlit.txt):**
```
streamlit>=1.28.0
plotly>=5.15.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=1.7.0
joblib>=1.3.0
```

### **Version Compatibility:**
- **Python:** 3.9, 3.10, 3.11, 3.12, 3.13
- **Streamlit:** 1.28+ (latest features used)
- **XGBoost:** 1.7+ (model compatibility)
- **Plotly:** 5.15+ (chart functionality)

---

## 🔧 Configuration & Customization

### **Application Configuration:**
```python
# streamlit_app.py - Page Configuration
st.set_page_config(
    page_title="PredSeeker - Developer Employment Predictor",
    page_icon="🎯", 
    layout="wide",
    initial_sidebar_state="expanded"
)
```

### **Model Parameters:**
```python
# XGBoost Model Configuration (from training)
{
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42
}
```

### **Skill Categories (Customizable):**
```python
SKILL_FAMILIES = {
    'Programming': [22 technologies],    # Add/remove languages
    'Web': [17 technologies],           # Add/remove frameworks  
    'Database': [13 technologies],      # Add/remove databases
    'CloudDevOps': [12 technologies]    # Add/remove cloud tools
}
```

### **UI Theme Customization:**
```css
/* Primary Colors */
--primary-color: #007bff;      /* Blue buttons, accents */
--header-color: #2E86AB;       /* Header text, titles */
--text-color: #212529;         /* Main text color */
--background-color: #ffffff;   /* App background */
--border-color: #dee2e6;       /* Form borders, dividers */
```

---

## 🚀 Performance Optimization

### **Model Loading Optimization:**
```python
@st.cache_resource
def load_model_and_info():
    """Cache model loading to prevent repeated file I/O"""
    # Loads once per session, significant performance improvement
```

### **CSS Optimization:**
- **Minified Styling:** Efficient CSS with minimal redundancy
- **Targeted Selectors:** Specific targeting to reduce style conflicts
- **Conditional Rendering:** UI elements load based on user state

### **Memory Management:**
- **Efficient DataFrames:** Minimal memory footprint for feature vectors
- **Cached Resources:** Model and metadata loaded once
- **Optimized Imports:** Only necessary libraries imported

### **Performance Benchmarks:**
- **App Startup:** < 3 seconds
- **Model Prediction:** < 1 second  
- **Chart Rendering:** < 0.5 seconds
- **Memory Usage:** ~150MB with model loaded

---

## 🔒 Security & Best Practices

### **Input Validation:**
```python
# Age validation
age = st.number_input(
    "Enter your age", 
    min_value=16,     # Minimum working age
    max_value=70,     # Reasonable maximum
    value=28
)

# Skill selection validation
if not selected_skills:
    st.warning("Please select at least one skill")
    return
```

### **Error Handling:**
```python
try:
    # Model loading with fallback
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    return None, None
```

### **Security Features:**
- ✅ **Input Sanitization:** All user inputs validated
- ✅ **File Path Security:** Safe model file loading
- ✅ **Error Boundaries:** Graceful error handling
- ✅ **No Sensitive Data:** No personal information stored

---

## 🌐 Deployment Options

### **1. Streamlit Cloud (Recommended for Sharing):**
```bash
# Push to GitHub repository
git add .
git commit -m "Deploy PredSeeker app"
git push origin main

# Deploy on Streamlit Cloud:
# 1. Visit streamlit.io/cloud
# 2. Connect GitHub repository
# 3. Select streamlit_app.py as main file
# 4. Deploy automatically
```

### **2. Docker Deployment:**
```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN pip install -r requirements_streamlit.txt

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### **3. Local Network Sharing:**
```bash
# Run with network access
streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8502

# Access from other devices on same network:
# http://[YOUR_IP_ADDRESS]:8502
```

### **4. Cloud Platform Deployment:**
- **AWS:** Elastic Beanstalk, EC2, or ECS
- **Google Cloud:** Cloud Run or App Engine
- **Azure:** Container Instances or App Service
- **Heroku:** Direct deployment with buildpack

---

## 🧪 Testing & Quality Assurance

### **Manual Testing Checklist:**
- ✅ **Model Loading:** Verify XGBoost model loads correctly
- ✅ **Skill Selection:** Test all 75+ technology checkboxes
- ✅ **Form Inputs:** Validate age, gender, education inputs
- ✅ **Predictions:** Ensure realistic probability outputs
- ✅ **Visualizations:** Check gauge and radar chart rendering
- ✅ **Responsive Design:** Test on different screen sizes
- ✅ **Error Handling:** Test with invalid inputs

### **Performance Testing:**
- **Load Time:** App should start within 3 seconds
- **Prediction Speed:** Results within 1 second
- **Memory Usage:** Monitor for memory leaks
- **Browser Compatibility:** Test across major browsers

### **Code Quality Standards:**
- **PEP 8 Compliance:** Python code formatting
- **Documentation:** Comprehensive docstrings
- **Error Handling:** Try-catch blocks for critical operations
- **Modular Design:** Reusable functions and components

---

## 📊 Monitoring & Analytics

### **Built-in Metrics:**
- **Model Performance:** ROC-AUC, Accuracy, Precision displayed
- **User Interaction:** Real-time skill selection feedback
- **Prediction Quality:** Reality check adjustments tracked
- **Error Logging:** Exception handling with user feedback

### **Future Monitoring Options:**
- **Usage Analytics:** Track user interactions and popular skills
- **Performance Metrics:** Response times and error rates  
- **Model Drift:** Monitor prediction accuracy over time
- **User Feedback:** Collect satisfaction and improvement suggestions

---

## 🔮 Future Enhancement Roadmap

### **Phase 2 Features:**
1. **User Authentication:** Profile management and history tracking
2. **Advanced Analytics:** Salary predictions and market trends
3. **API Development:** REST endpoints for integration
4. **Mobile App:** Native iOS/Android applications

### **Technical Improvements:**
1. **Model Updates:** Retrain with recent data
2. **A/B Testing:** Test different UI layouts
3. **Performance:** Further optimize loading and rendering
4. **Accessibility:** Enhanced screen reader support

### **Business Features:**
1. **Learning Paths:** Skill development recommendations
2. **Industry Insights:** Sector-specific employment trends
3. **Networking:** Connect with similar developers
4. **Progress Tracking:** Career development monitoring

---

## 📝 Troubleshooting Guide

### **Common Issues & Solutions:**

#### **1. Model Loading Errors:**
```bash
# Error: Model file not found
# Solution: Ensure models/ directory exists with trained model
python -c "import os; print(os.path.exists('models/best_employment_model.joblib'))"
```

#### **2. Streamlit Port Conflicts:**
```bash
# Error: Port 8501 already in use
# Solution: Use different port
streamlit run streamlit_app.py --server.port 8502
```

#### **3. Dependency Conflicts:**
```bash  
# Error: Package version conflicts
# Solution: Create fresh virtual environment
pip freeze > current_requirements.txt
pip uninstall -r current_requirements.txt -y
pip install -r requirements_streamlit.txt
```

#### **4. CSS Styling Issues:**
```python
# Error: UI elements not styled correctly
# Solution: Clear browser cache or use incognito mode
# CSS changes sometimes require cache refresh
```

### **Debug Mode:**
```bash
# Run with debug information
streamlit run streamlit_app.py --logger.level debug
```

---

## 🤝 Contributing & Development

### **Development Workflow:**
1. **Fork Repository:** Create personal copy for changes
2. **Create Branch:** Feature-specific development branches
3. **Make Changes:** Follow coding standards and documentation
4. **Test Thoroughly:** Manual testing across browsers/devices
5. **Submit PR:** Pull request with detailed description

### **Code Standards:**
- **Python:** Follow PEP 8 style guidelines
- **Documentation:** Comprehensive docstrings and comments
- **Testing:** Manual testing checklist for all changes
- **Git:** Descriptive commit messages and logical commits

### **Enhancement Ideas:**
- Additional skill categories (Mobile, AI/ML, etc.)
- Industry-specific models (Finance, Healthcare, etc.)
- Multi-language support for global users
- Integration with job boards and career platforms

---

## 📞 Support & Contact

### **Technical Support:**
- **Issues:** GitHub Issues for bug reports
- **Documentation:** Comprehensive README and guides
- **Community:** Streamlit community forums for framework questions

### **Project Information:**
- **Repository:** https://github.com/ChanakaWickramasingha/predSeeker
- **Branch:** Thiyangi_training (development branch)
- **License:** MIT (check repository for current license)

---

*Technology Stack Guide - October 1, 2025*
*PredSeeker Project - Complete Implementation*