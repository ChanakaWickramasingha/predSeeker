"""
Data Preprocessing Module for Stack Overflow Dev        print(f"[OK] Dataset loaded successfully: {df.shape}")loper Survey Analysis

This module contains functions for cleaning and preprocessing the developer survey data,
with special focus on the HaveWorkedWith column processing.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Define skill families for technology categorization
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

def load_data(file_path):
    """
    Load the Stack Overflow survey dataset
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Dataset loaded successfully: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"[ERROR] File not found: {file_path}")
        return None
    except Exception as e:
        print(f"[ERROR] Error loading dataset: {e}")
        return None

def parse_technologies(tech_string):
    """
    Parse the HaveWorkedWith column into a list of technologies
    
    Args:
        tech_string (str): Semicolon-separated technology string
        
    Returns:
        list: List of individual technologies
    """
    if pd.isna(tech_string) or tech_string == '' or tech_string == 'Unknown':
        return []
    
    # Split by semicolon and clean up
    techs = [tech.strip() for tech in str(tech_string).split(';') if tech.strip()]
    return techs

def calculate_skill_family_scores(tech_list):
    """
    Calculate percentage scores for each skill family
    
    Args:
        tech_list (list): List of technologies
        
    Returns:
        tuple: (scores_dict, binary_flags_dict)
    """
    scores = {}
    binary_flags = {}
    
    for family_name, family_techs in SKILL_FAMILIES.items():
        # Count technologies from this family
        known_techs = [tech for tech in tech_list if tech in family_techs]
        known_count = len(known_techs)
        total_count = len(family_techs)
        
        # Calculate percentage score
        percentage = (known_count / total_count) * 100 if total_count > 0 else 0
        
        # Store results
        scores[f'{family_name}_Score'] = round(percentage, 1)
        binary_flags[f'Has_{family_name}'] = 1 if known_count > 0 else 0
    
    return scores, binary_flags

def calculate_derived_features(binary_flags):
    """
    Calculate skill breadth and full-stack indicator
    
    Args:
        binary_flags (dict): Dictionary of binary skill flags
        
    Returns:
        dict: Derived features
    """
    skill_breadth = sum(binary_flags.values())
    
    is_fullstack = 1 if (binary_flags['Has_Programming'] == 1 and 
                        binary_flags['Has_Web'] == 1 and 
                        binary_flags['Has_Database'] == 1) else 0
    
    return {
        'Skill_Breadth': skill_breadth,
        'Is_FullStack': is_fullstack
    }

def process_haveworkedwith_column(df):
    """
    Process the HaveWorkedWith column and create the 10 essential features
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with new technology features
    """
    print("=== PROCESSING HAVEWORKEDWITH COLUMN ===")
    
    # Create a copy to work with
    df_processed = df.copy()
    
    # Parse technologies for each row
    print("Parsing technologies...")
    df_processed['Technologies_List'] = df_processed['HaveWorkedWith'].apply(parse_technologies)
    
    # Initialize lists to store results
    all_scores = []
    all_binary_flags = []
    all_derived = []
    
    # Process each row
    for idx, tech_list in enumerate(df_processed['Technologies_List']):
        # Calculate scores and flags
        scores, binary_flags = calculate_skill_family_scores(tech_list)
        derived = calculate_derived_features(binary_flags)
        
        all_scores.append(scores)
        all_binary_flags.append(binary_flags)
        all_derived.append(derived)
    
    # Convert to DataFrames and merge
    scores_df = pd.DataFrame(all_scores)
    binary_df = pd.DataFrame(all_binary_flags)
    derived_df = pd.DataFrame(all_derived)
    
    # Add new columns to main dataframe
    df_processed = pd.concat([df_processed, scores_df, binary_df, derived_df], axis=1)
    
    print("[OK] Technology processing completed!")
    print(f"Added {len(scores_df.columns) + len(binary_df.columns) + len(derived_df.columns)} new columns")
    
    return df_processed

def handle_missing_values(df_processed):
    """
    Handle missing values in all columns
    
    Args:
        df_processed (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with missing values handled
    """
    print("=== HANDLING MISSING VALUES ===")
    
    # Check missing values before
    print("Missing values before cleaning:")
    missing_before = df_processed.isnull().sum()
    print(missing_before[missing_before > 0])
    
    # Handle missing values by column type
    
    # Categorical columns - fill with mode
    categorical_cols = ['Age', 'Accessibility', 'EdLevel', 'Gender', 'MentalHealth', 'MainBranch', 'Country']
    for col in categorical_cols:
        if col in df_processed.columns:
            mode_value = df_processed[col].mode()[0] if not df_processed[col].mode().empty else 'Unknown'
            df_processed[col] = df_processed[col].fillna(mode_value)
            print(f"[OK] {col}: Filled with '{mode_value}'")
    
    # Numerical columns - fill with median
    numerical_cols = ['YearsCode', 'YearsCodePro', 'PreviousSalary']
    for col in numerical_cols:
        if col in df_processed.columns:
            median_value = df_processed[col].median()
            df_processed[col] = df_processed[col].fillna(median_value)
            print(f"[OK] {col}: Filled with {median_value}")
    
    # ComputerSkills - special handling
    if 'ComputerSkills' in df_processed.columns:
        # If ComputerSkills is missing, use the count from parsed technologies
        df_processed['Parsed_Tech_Count'] = df_processed['Technologies_List'].apply(len)
        df_processed['ComputerSkills'] = df_processed['ComputerSkills'].fillna(df_processed['Parsed_Tech_Count'])
        print("[OK] ComputerSkills: Filled with parsed technology count where missing")
    
    # Check missing values after
    print("\nMissing values after cleaning:")
    missing_after = df_processed.isnull().sum()
    print(missing_after[missing_after > 0])
    
    return df_processed

def create_additional_features(df_processed):
    """
    Create additional features for employment prediction
    
    Args:
        df_processed (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with additional features
    """
    print("=== CREATING ADDITIONAL FEATURES ===")
    
    # Age group (binary)
    df_processed['IsYoung'] = (df_processed['Age'] == '<35').astype(int)
    print("[OK] IsYoung: Age group feature")
    
    # Education level (ordinal)
    education_order = {
        'NoHigherEd': 0, 'Other': 1, 'Undergraduate': 2, 'Master': 3, 'PhD': 4
    }
    df_processed['EducationLevel_Numeric'] = df_processed['EdLevel'].map(education_order).fillna(1)
    print("[OK] EducationLevel_Numeric: Ordinal education encoding")
    
    # Developer status
    df_processed['IsDeveloper'] = (df_processed['MainBranch'] == 'Dev').astype(int)
    print("[OK] IsDeveloper: Developer role indicator")
    
    # Mental health
    df_processed['HasMentalHealthConcerns'] = (df_processed['MentalHealth'] == 'Yes').astype(int)
    print("[OK] HasMentalHealthConcerns: Mental health indicator")
    
    # Experience features
    df_processed['HasCodingExperience'] = (df_processed['YearsCode'] > 0).astype(int)
    df_processed['HasProfessionalExperience'] = (df_processed['YearsCodePro'] > 0).astype(int)
    df_processed['ExperienceRatio'] = df_processed['YearsCodePro'] / (df_processed['YearsCode'] + 1)
    print("[OK] Experience features: Coding and professional experience indicators")
    
    # Salary features
    df_processed['HasSalaryInfo'] = (~df_processed['PreviousSalary'].isna()).astype(int)
    print("[OK] HasSalaryInfo: Salary information availability")
    
    return df_processed

def select_final_features(df_processed):
    """
    Select the final set of features for the employment prediction model
    
    Args:
        df_processed (pd.DataFrame): Processed dataframe
        
    Returns:
        tuple: (X, y, feature_names)
    """
    print("=== SELECTING FINAL FEATURES ===")
    
    # The 10 core technology features
    tech_features = [
        'Programming_Score', 'Web_Score', 'Database_Score', 'CloudDevOps_Score',
        'Has_Programming', 'Has_Web', 'Has_Database', 'Has_CloudDevOps',
        'Skill_Breadth', 'Is_FullStack'
    ]
    
    # Additional important features
    additional_features = [
        'YearsCode', 'YearsCodePro', 'ComputerSkills',
        'EducationLevel_Numeric', 'IsYoung', 'IsDeveloper',
        'HasCodingExperience', 'HasProfessionalExperience', 'ExperienceRatio',
        'HasSalaryInfo'
    ]
    
    # Combine all features
    selected_features = tech_features + additional_features
    target_column = 'Employed'
    
    # Create final dataset
    X = df_processed[selected_features].copy()
    y = df_processed[target_column].copy()
    
    print(f"[OK] Selected {len(selected_features)} features:")
    print("Technology Features (10):")
    for feature in tech_features:
        print(f"   - {feature}")
    print("Additional Features (10):")
    for feature in additional_features:
        print(f"   - {feature}")
    
    return X, y, selected_features

def prepare_final_dataset(X, y):
    """
    Scale features and split data for machine learning
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target variable
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler)
    """
    print("=== PREPARING FINAL DATASET ===")
    
    # Handle any remaining missing or infinite values
    X = X.fillna(X.median())
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale numerical features
    numerical_features = X.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    X_test_scaled[numerical_features] = scaler.transform(X_test[numerical_features])
    
    print(f"[OK] Dataset split:")
    print(f"   Training set: {X_train_scaled.shape}")
    print(f"   Test set: {X_test_scaled.shape}")
    print(f"   Features: {X.shape[1]}")
    
    print(f"\n[OK] Target distribution in training set:")
    print(y_train.value_counts(normalize=True) * 100)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def main_preprocessing_pipeline(file_path):
    """
    Complete preprocessing pipeline
    
    Args:
        file_path (str): Path to the raw dataset
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler, df_processed)
    """
    print("STARTING PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Step 1: Load data
    df = load_data(file_path)
    if df is None:
        return None
    
    # Step 2: Process HaveWorkedWith column
    df_processed = process_haveworkedwith_column(df)
    
    # Step 3: Handle missing values
    df_processed = handle_missing_values(df_processed)
    
    # Step 4: Create additional features
    df_processed = create_additional_features(df_processed)
    
    # Step 5: Select final features
    X, y, feature_names = select_final_features(df_processed)
    
    # Step 6: Prepare final dataset
    X_train, X_test, y_train, y_test, scaler = prepare_final_dataset(X, y)
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Original dataset: {df.shape}")
    print(f"Processed dataset: {X.shape}")
    print(f"New technology features: 10")
    print(f"Total features for modeling: {X.shape[1]}")
    print(f"Ready for machine learning!")
    
    return X_train, X_test, y_train, y_test, scaler, df_processed

if __name__ == "__main__":
    # Example usage
    file_path = "data/raw/stackoverflow_with_nulls.csv"
    result = main_preprocessing_pipeline(file_path)
    
    if result:
        X_train, X_test, y_train, y_test, scaler, df_processed = result
        
        # Save processed data
        df_processed.to_csv("data/processed/preprocessed_data.csv", index=False)
        print("\n[OK] Processed data saved to data/processed/preprocessed_data.csv")