"""
Data Preprocessing Module for Stack Overflow Developer Survey Analysis

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
    """Load the Stack Overflow survey dataset"""
    try:
        df = pd.read_csv(file_path)
        print(f"[OK] Dataset loaded successfully: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"[ERROR] File not found: {file_path}")
        return None
    except Exception as e:
        print(f"[ERROR] Error loading dataset: {e}")
        return None

def parse_technologies(tech_string):
    """Parse the HaveWorkedWith column into a list of technologies"""
    if pd.isna(tech_string) or tech_string == '' or tech_string == 'Unknown':
        return []

    # Split by semicolon and clean up
    techs = [tech.strip() for tech in str(tech_string).split(';') if tech.strip()]
    return techs

def calculate_skill_family_scores(tech_list):
    """Calculate percentage scores for each skill family"""
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
    """Calculate skill breadth and full-stack indicator"""
    skill_breadth = sum(binary_flags.values())

    is_fullstack = 1 if (binary_flags['Has_Programming'] == 1 and
                        binary_flags['Has_Web'] == 1 and
                        binary_flags['Has_Database'] == 1) else 0

    return {
        'Skill_Breadth': skill_breadth,
        'Is_FullStack': is_fullstack
    }

def process_haveworkedwith_column(df):
    """Process the HaveWorkedWith column and create the 10 essential features"""
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
    """Handle missing values in all columns"""
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
    """Create additional features for employment prediction"""
    print("=== CREATING ADDITIONAL FEATURES ===")

    # Age group (binary) - KEEPING despite low correlation as it's interpretable
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

    # Accessibility needs (binary encoding)
    if 'Accessibility' in df_processed.columns:
        df_processed['HasAccessibilityNeeds'] = (df_processed['Accessibility'] == 'Yes').astype(int)
        print(f"[OK] HasAccessibilityNeeds: {(df_processed['Accessibility'] == 'Yes').sum()} with accessibility needs")

    # Gender (one-hot encoding)
    if 'Gender' in df_processed.columns:
        gender_counts = df_processed['Gender'].value_counts()
        print(f"[INFO] Gender distribution: {gender_counts.to_dict()}")

        # Create one-hot encoded gender features
        df_processed['Gender_Man'] = (df_processed['Gender'] == 'Man').astype(int)
        df_processed['Gender_Woman'] = (df_processed['Gender'] == 'Woman').astype(int)
        df_processed['Gender_NonBinary'] = (df_processed['Gender'] == 'NonBinary').astype(int)

        man_count = df_processed['Gender_Man'].sum()
        woman_count = df_processed['Gender_Woman'].sum()
        nonbinary_count = df_processed['Gender_NonBinary'].sum()

        print(f"[OK] Gender one-hot encoded:")
        print(f"   - Gender_Man: {man_count:,} ({man_count/len(df_processed)*100:.1f}%)")
        print(f"   - Gender_Woman: {woman_count:,} ({woman_count/len(df_processed)*100:.1f}%)")
        print(f"   - Gender_NonBinary: {nonbinary_count:,} ({nonbinary_count/len(df_processed)*100:.1f}%)")

    # Experience features - KEEPING HasProfessionalExperience as requested
    df_processed['HasProfessionalExperience'] = (df_processed['YearsCodePro'] > 0).astype(int)
    print(f"[OK] HasProfessionalExperience: {df_processed['HasProfessionalExperience'].sum():,} with professional experience")

    # Note: Still dropping HasCodingExperience and ExperienceRatio due to very low correlation
    # df_processed['HasCodingExperience'] = (df_processed['YearsCode'] > 0).astype(int)  # DROPPED: Low correlation (0.0171)
    # df_processed['ExperienceRatio'] = df_processed['YearsCodePro'] / (df_processed['YearsCode'] + 1)  # DROPPED: Low correlation (0.0145)
    print("[INFO] Kept HasProfessionalExperience, still dropped HasCodingExperience and ExperienceRatio")

    # Salary features
    df_processed['HasSalaryInfo'] = (~df_processed['PreviousSalary'].isna()).astype(int)
    print("[OK] HasSalaryInfo: Salary information availability")

    return df_processed

def select_final_features(df_processed):
    """Select the final set of features for the employment prediction model"""
    print("=== SELECTING FINAL FEATURES ===")

    # The core technology features - KEEPING Has_Programming as requested
    tech_features = [
        'Programming_Score', 'Web_Score', 'Database_Score', 'CloudDevOps_Score',
        'Has_Programming', 'Has_Web', 'Has_Database', 'Has_CloudDevOps',  # Added back Has_Programming
        'Skill_Breadth', 'Is_FullStack'
    ]

    # Additional important features - ADDED BACK HasProfessionalExperience
    additional_features = [
        'ComputerSkills',  # Strong predictor (0.5838 correlation)
        'EducationLevel_Numeric', 'IsYoung', 'IsDeveloper',
        'HasMentalHealthConcerns', 'HasAccessibilityNeeds',
        'Gender_Man', 'Gender_Woman', 'Gender_NonBinary',  # Gender one-hot encoded
        'HasProfessionalExperience',  # Added back as requested (even with low correlation)
        'HasSalaryInfo'
        # STILL REMOVED: YearsCode (0.0020), YearsCodePro (0.0054), HasCodingExperience, ExperienceRatio
    ]

    # Combine all features
    selected_features = tech_features + additional_features
    target_column = 'Employed'

    # Validate features exist
    missing_features = [f for f in selected_features if f not in df_processed.columns]
    if missing_features:
        print(f"[WARNING] Missing features: {missing_features}")
        selected_features = [f for f in selected_features if f in df_processed.columns]

    # Create final dataset
    X = df_processed[selected_features].copy()
    y = df_processed[target_column].copy()

    print(f"[OK] Selected {len(selected_features)} features (OPTIMIZED - removed low predictive features):")
    print(f"Technology Features ({len(tech_features)}):")
    for feature in tech_features:
        print(f"   - {feature}")
    print(f"Additional Features ({len(additional_features)}):")
    for feature in additional_features:
        print(f"   - {feature}")

    # Show gender encoding specifically
    gender_features = [f for f in selected_features if f.startswith('Gender_')]
    if gender_features:
        print(f"Gender Features ({len(gender_features)}):")
        for feature in gender_features:
            print(f"   - {feature}")

    # Show features that were dropped - UPDATED to reflect kept features
    dropped_features = [
        'YearsCode (corr: 0.0020)', 'YearsCodePro (corr: 0.0054)',
        'ExperienceRatio (corr: 0.0145)', 'HasCodingExperience (corr: 0.0171)'
    ]
    kept_low_corr_features = [
        'HasProfessionalExperience (corr: 0.0234)', 'Has_Programming (corr: 0.0484)'
    ]
    print(f"\n[INFO] Dropped {len(dropped_features)} low-predictive features:")
    for feature in dropped_features:
        print(f"   - {feature}")
    print(f"\n[INFO] Kept {len(kept_low_corr_features)} low-correlation features (as requested):")
    for feature in kept_low_corr_features:
        print(f"   - {feature}")

    return X, y, selected_features

def prepare_final_dataset(X, y):
    """Scale features and split data for machine learning"""
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

def drop_unwanted_features(df_processed):
    """Drop features with low predictive capacity and unwanted columns"""
    print("=== DROPPING UNWANTED FEATURES ===")

    # Define features to drop
    features_to_drop = [
        # Low predictive capacity (correlation < 0.05) - KEEPING some as requested
        'YearsCode',           # 0.0020 correlation
        'YearsCodePro',        # 0.0054 correlation
        'ExperienceRatio',     # 0.0145 correlation
        'HasCodingExperience', # 0.0171 correlation
        # 'HasProfessionalExperience', # 0.0234 correlation - KEEPING as requested
        # 'Has_Programming',     # 0.0484 correlation - KEEPING as requested

        # Redundant/problematic features
        'Unnamed: 0',         # Index column
        'Employment',         # Target variable (data leakage)
        'Technologies_List',  # Text - already processed
        'HaveWorkedWith',     # Text - already processed
        'Country',            # High cardinality (172 countries)

        # Original categorical columns (replaced with encoded versions)
        'Age',                # Replaced with IsYoung
        'Accessibility',      # Replaced with HasAccessibilityNeeds
        'EdLevel',            # Replaced with EducationLevel_Numeric
        'Gender',             # Replaced with Gender_Man, Gender_Woman, Gender_NonBinary
        'MentalHealth',       # Replaced with HasMentalHealthConcerns
        'MainBranch',         # Replaced with IsDeveloper

        # Parsed data column
        'Parsed_Tech_Count'   # Helper column
    ]

    # Drop features that exist in the dataset
    initial_shape = df_processed.shape
    columns_to_drop = [col for col in features_to_drop if col in df_processed.columns]

    if columns_to_drop:
        df_processed = df_processed.drop(columns=columns_to_drop)
        print(f"[OK] Dropped {len(columns_to_drop)} unwanted features:")
        for col in columns_to_drop:
            print(f"   - {col}")

    print(f"[INFO] Dataset shape: {initial_shape} -> {df_processed.shape}")
    print(f"[INFO] Removed {initial_shape[1] - df_processed.shape[1]} columns")

    return df_processed

def main_preprocessing_pipeline(file_path):
    """Complete preprocessing pipeline with feature optimization"""
    print("STARTING OPTIMIZED PREPROCESSING PIPELINE")
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

    # Step 5: Drop unwanted features
    df_processed = drop_unwanted_features(df_processed)

    # Step 6: Select final features for ML
    X, y, feature_names = select_final_features(df_processed)

    # Step 7: Prepare final dataset
    X_train, X_test, y_train, y_test, scaler = prepare_final_dataset(X, y)

    print("\n" + "="*60)
    print("OPTIMIZED PREPROCESSING COMPLETED!")
    print("="*60)
    print(f"Original dataset: {df.shape}")
    print(f"Clean processed dataset: {df_processed.shape}")
    print(f"ML feature set: {X.shape}")
    print(f"Features selected for modeling: {X.shape[1]} (optimized)")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print("Ready for machine learning with optimized features!")

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
