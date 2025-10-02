# PredSeeker - AI-Powered Developer Employment Predictor

## Stack Overflow Developer Survey Analysis & Machine Learning Application

PredSeeker is a comprehensive machine learning project that analyzes Stack Overflow developer survey data to predict employment status based on technical skills, demographics, and professional experience. The project includes both data science analysis and a production-ready web application.

## Project Overview

This project transforms raw developer survey data into actionable employment predictions through advanced feature engineering, machine learning model training, and an intuitive web interface built with Streamlit.

## Project Structure

```
predSeeker/
├── data/                           # Data storage and processing
│   ├── raw/                       # Original Stack Overflow survey data
│   └── processed/                 # Cleaned and engineered datasets
│       ├── preprocessed_data_clean.csv
│       ├── X_train.csv, X_test.csv
│       └── y_train.csv, y_test.csv
├── src/                           # Core Python modules
│   ├── preprocessing_clean.py     # Advanced data preprocessing
│   ├── feature_engineering.py    # Feature creation and selection
│   ├── eda.py                    # Exploratory data analysis
│   └── models.py                 # Machine learning pipeline
├── notebooks/                     # Jupyter analysis notebooks
│   ├── 01_EDA.ipynb              # Comprehensive data exploration
│   └── 02_Model_Training.ipynb   # Model training and evaluation
├── models/                        # Trained model artifacts
│   ├── best_employment_model.joblib
│   ├── model_info.json
│   └── model_comparison.csv
├── results/                       # Documentation and analysis
│   ├── FINAL_PROJECT_SUMMARY.md
│   ├── STREAMLIT_APP_SUMMARY.md
│   ├── TECH_STACK_GUIDE.md
│   └── preprocessing_summaries/
├── streamlit_app.py              # Production web application
├── requirements.txt              # Core dependencies
├── requirements_streamlit.txt    # Web app dependencies
└── README.md                     # Project documentation
```

## Key Features

### Advanced Data Processing
- Strategic missing value handling with domain knowledge
- Sophisticated parsing of semicolon-separated technology lists
- Creation of 21 engineered features from raw survey data
- Comprehensive skill family categorization and scoring

### Technology Skill Analysis
The project processes over 75 technologies across four major categories:

**Programming Languages & Frameworks**
- Python, Java, JavaScript, C++, C#, PHP, Ruby, Go, Rust, Swift, Kotlin, and more
- Calculates Programming_Score as percentage of known languages

**Web Technologies**
- HTML/CSS, React.js, Angular, Vue.js, Node.js, Express.js, Django, Flask
- Generates Web_Score based on frontend and backend technology knowledge

**Database Systems**
- MySQL, PostgreSQL, MongoDB, Redis, SQLite, Oracle, SQL Server
- Creates Database_Score reflecting data management capabilities

**Cloud & DevOps Tools**
- AWS, Azure, Google Cloud, Docker, Kubernetes, Git, Jenkins, Terraform
- Produces CloudDevOps_Score indicating modern development practices

### Machine Learning Pipeline
- Binary classification for employment prediction
- Comparison of 5 different algorithms: Logistic Regression, Random Forest, XGBoost, SVM, Extra Trees
- Comprehensive model evaluation with ROC-AUC, accuracy, precision, recall, and F1-score
- Feature importance analysis and cross-validation
- Best model selection based on performance metrics

### Web Application
- Interactive Streamlit interface with modern UI design
- Real-time employment probability predictions
- Skill selection through intuitive checkbox interface
- Dynamic visualization with confidence gauges and radar charts
- Personalized career recommendations
- Responsive design for desktop and mobile devices

## Technical Methodology

### Feature Engineering Process
The core innovation lies in transforming the HaveWorkedWith column into meaningful predictive features:

```python
# Input: "Python;React.js;PostgreSQL;Docker;AWS"
# Output Features:
Programming_Score: 4.5%        # 1 out of 22 programming languages
Web_Score: 5.9%               # 1 out of 17 web technologies  
Database_Score: 8.3%          # 1 out of 12 database systems
CloudDevOps_Score: 16.7%      # 2 out of 12 cloud/devops tools
Has_Programming: 1            # Binary indicator
Has_Web: 1                    # Binary indicator
Has_Database: 1               # Binary indicator
Has_CloudDevOps: 1           # Binary indicator
Skill_Breadth: 4             # Number of skill categories
Is_FullStack: 1              # Has programming + web + database
ComputerSkills: 5            # Total number of technologies
```

### Model Performance
The trained XGBoost model achieves excellent performance metrics:

- **ROC-AUC Score**: 0.8951 (89.51%)
- **Accuracy**: 0.8110 (81.10%)
- **F1-Score**: 0.8280 (82.80%)
- **Precision**: 0.8090 (80.90%)
- **Recall**: 0.8480 (84.80%)

### Dataset Statistics
- **Training Samples**: 58,769 developers
- **Test Samples**: 14,693 developers
- **Features**: 21 engineered features
- **Class Distribution**: 54% employed, 46% unemployed
- **Technologies Analyzed**: 75+ different tools and languages

## Getting Started

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 8GB+ RAM recommended for full dataset processing

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/predSeeker.git
   cd predSeeker
   ```

2. **Set Up Environment**
   ```bash
   # Create virtual environment (recommended)
   python -m venv predseeker_env
   source predseeker_env/bin/activate  # On Windows: predseeker_env\Scripts\activate
   
   # Install core dependencies
   pip install -r requirements.txt
   
   # Install Streamlit app dependencies
   pip install -r requirements_streamlit.txt
   ```

3. **Prepare Dataset**
   - Download Stack Overflow Developer Survey data
   - Place the CSV file in `data/raw/` directory
   - Ensure the file contains required columns: HaveWorkedWith, Employment, Age, Gender, etc.

### Running the Application

#### Web Application (Recommended)
```bash
streamlit run streamlit_app.py
```
Navigate to `http://localhost:8501` to access the interactive employment predictor.

#### Jupyter Analysis
```bash
jupyter notebook
```
Open and run notebooks in sequence:
1. `notebooks/01_EDA.ipynb` - Data exploration and visualization
2. `notebooks/02_Model_Training.ipynb` - Model training and evaluation

#### Python Scripts
```bash
# Run preprocessing
python src/preprocessing_clean.py

# Train models
python src/models.py
```

## Usage Examples

### Employment Prediction via Web App
1. Launch the Streamlit application
2. Select your technical skills across four categories
3. Enter demographic and professional information
4. Click "Predict Employment Probability"
5. View results with confidence score and personalized recommendations

### Model Training and Evaluation
```python
from src.models import train_employment_models
from src.preprocessing_clean import preprocess_survey_data

# Load and preprocess data
df_processed = preprocess_survey_data('data/raw/survey_data.csv')

# Train multiple models
results = train_employment_models(df_processed)

# Get best model
best_model = results['best_model']
performance_metrics = results['metrics']
```

## Model Architecture

### Feature Categories
1. **Technical Skills**: Programming, Web, Database, CloudDevOps scores
2. **Skill Indicators**: Binary flags for each technology category
3. **Derived Features**: Skill breadth, full-stack indicator, computer skills count
4. **Demographics**: Age group, gender, accessibility needs, mental health
5. **Professional**: Education level, developer type, experience, salary transparency

### Algorithm Comparison
The project evaluates multiple machine learning algorithms:

- **XGBoost**: Best overall performance with 89.51% ROC-AUC
- **Random Forest**: Strong ensemble method with good interpretability
- **Logistic Regression**: Fast baseline with linear decision boundaries
- **Support Vector Machine**: Non-linear classification with RBF kernel
- **Extra Trees**: Alternative ensemble with randomized splits

## Results and Insights

### Key Findings
- Programming language diversity strongly correlates with employment success
- Full-stack developers (programming + web + database skills) show higher employment rates
- Cloud and DevOps skills are increasingly important for employment
- Professional experience and education level remain significant predictors
- Age demographics show interesting employment pattern variations

### Feature Importance Ranking
1. Programming_Score: Technical programming capability
2. ComputerSkills: Total number of technologies known
3. HasProfessionalExperience: Prior work experience indicator
4. Web_Score: Web development technology knowledge
5. Skill_Breadth: Diversity across technology categories

## Deployment Options

### Local Development
- Streamlit development server for testing and development
- Jupyter notebooks for data analysis and model experimentation

### Production Deployment
- **Streamlit Cloud**: Easy deployment with GitHub integration
- **Heroku**: Container-based deployment with custom buildpacks
- **AWS/GCP/Azure**: Cloud platform deployment with scaling capabilities
- **Docker**: Containerized deployment for consistent environments

## Performance Optimization

### Data Processing
- Efficient pandas operations with vectorization
- Memory-optimized feature engineering
- Intelligent data type selection to reduce memory footprint

### Model Training
- Stratified sampling for balanced training sets
- Cross-validation with appropriate fold selection
- Hyperparameter tuning for optimal performance
- Model serialization for fast loading in production

### Web Application
- Caching of model loading for faster startup
- Optimized CSS and JavaScript for responsive UI
- Progressive loading of charts and visualizations

## Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Install development dependencies
4. Make changes with appropriate tests
5. Submit a pull request with detailed description

### Code Standards
- Follow PEP 8 style guidelines
- Include docstrings for all functions and classes
- Add type hints where appropriate
- Write unit tests for new functionality
- Update documentation for significant changes

## Dependencies

### Core Data Science
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing and array operations
- **scikit-learn**: Machine learning algorithms and metrics
- **xgboost**: Gradient boosting framework
- **matplotlib/seaborn**: Data visualization and plotting

### Web Application
- **streamlit**: Interactive web application framework
- **plotly**: Interactive charts and visualizations
- **joblib**: Model serialization and loading

### Development Tools
- **jupyter**: Interactive development environment
- **pytest**: Testing framework
- **black**: Code formatting
- **flake8**: Code linting

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Acknowledgments

- Stack Overflow for providing comprehensive developer survey data
- The open-source community for excellent Python libraries
- Academic research in employment prediction and skill analysis
- Contributors and users who provide feedback and improvements

## Contact

**Project Team**: PredSeeker Development Team  
**Course**: IT3051 - Fundamentals of Data Mining  
**Institution**: University of Kelaniya  
**Year**: 2025

For questions, suggestions, or contributions, please open an issue on GitHub or contact the development team.

---

**Note**: This project is for educational and research purposes. Employment predictions should be used as guidance alongside other career planning resources.