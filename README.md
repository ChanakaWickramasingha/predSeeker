# FDM Data Mining Project

## Stack Overflow Developer Survey Analysis & Employment Prediction

This project analyzes Stack Overflow developer survey data to predict employment status based on skills, experience, and demographics.

## Project Structure

```
FDMproject/
├── data/                    # Raw and processed datasets
│   ├── raw/                # Original survey data
│   └── processed/          # Cleaned and feature-engineered data
├── src/                    # Source code modules
│   ├── preprocessing.py    # Data preprocessing functions
│   ├── feature_engineering.py  # Feature creation from HaveWorkedWith
│   ├── eda.py             # Exploratory data analysis
│   └── models.py          # Machine learning models
├── notebooks/             # Jupyter notebooks for analysis
│   ├── 01_EDA.ipynb       # Exploratory Data Analysis
│   ├── 02_Preprocessing.ipynb  # Data preprocessing
│   └── 03_Modeling.ipynb  # Model training and evaluation
├── models/                # Saved model files
├── results/               # Analysis results and visualizations
└── requirements.txt       # Python dependencies
```

## Key Features

### Data Preprocessing
- Handle missing values strategically
- Parse HaveWorkedWith column into skill families
- Create 10 essential technology features:
  - Programming_Score, Web_Score, Database_Score, CloudDevOps_Score
  - Has_Programming, Has_Web, Has_Database, Has_CloudDevOps
  - Skill_Breadth, Is_FullStack

### Skill Family Analysis
- **Programming Languages**: Python, Java, JavaScript, C++, etc.
- **Web Technologies**: HTML/CSS, React.js, Angular, Node.js, etc.
- **Databases**: MySQL, PostgreSQL, MongoDB, etc.
- **Cloud/DevOps**: AWS, Docker, Kubernetes, Git, etc.

### Employment Prediction
- Binary classification: Employed vs Not Employed
- Features based on technical skills, experience, and education
- Multiple ML algorithms for comparison

## Getting Started

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Place Dataset**
   - Add `stackoverflow_with_nulls.csv` to `data/raw/`

3. **Run Analysis**
   - Start with `notebooks/01_EDA.ipynb` for exploration
   - Use `notebooks/02_Preprocessing.ipynb` for data preparation
   - Train models with `notebooks/03_Modeling.ipynb`

## Key Methodology

### HaveWorkedWith Column Processing
The main innovation is processing the semicolon-separated technology list into meaningful features:

```python
# Example: "Python;React.js;PostgreSQL;Docker;Git"
# Becomes:
- Programming_Score: 4.5% (1/22 programming languages)
- Web_Score: 5.9% (1/17 web technologies)  
- Database_Score: 8.3% (1/12 databases)
- CloudDevOps_Score: 16.7% (2/12 cloud/devops tools)
- Is_FullStack: 1 (has programming + web + database)
```

### Employment Prediction Target
- **Target Variable**: `Employed` (1 = Employed, 0 = Not Employed)
- **Primary Features**: Technology skills, experience, education
- **Goal**: Predict employment likelihood based on developer profile

## Results Expected
- Comprehensive EDA of developer skills landscape
- Engineered features from raw technology data
- Trained ML models for employment prediction
- Insights into which skills correlate with employment success

## Dependencies
- pandas, numpy: Data manipulation
- scikit-learn: Machine learning
- matplotlib, seaborn: Visualization
- jupyter: Interactive analysis

---

**Author**: FDM Data Mining Team  
**Course**: IT3051 - Fundamentals of Data Mining  
**Year**: 2025