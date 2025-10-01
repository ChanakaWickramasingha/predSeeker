"""
Exploratory Data Analysis (EDA) Module for Stack Overflow Developer Survey Analysis

This module contains functions for comprehensive exploratory data analysis
including visualizations and statistical summaries.
"""

import pandas as pd
import numpy as np
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("⚠️ Visualization libraries not available. Install matplotlib and seaborn for full functionality.")

def basic_dataset_overview(df):
    """
    Provide basic overview of the dataset

    Args:
        df (pd.DataFrame): Input dataset

    Returns:
        dict: Overview statistics
    """
    print("=== BASIC DATASET OVERVIEW ===")

    overview = {
        'shape': df.shape,
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
        'columns': list(df.columns),
        'dtypes': df.dtypes.value_counts().to_dict(),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum()
    }

    print(f"Dataset Shape: {overview['shape']}")
    print(f"Memory Usage: {overview['memory_usage']:.2f} MB")
    print(f"Total Columns: {len(overview['columns'])}")
    print(f"Data Types: {overview['dtypes']}")
    print(f"Missing Values: {overview['missing_values']}")
    print(f"Duplicate Rows: {overview['duplicate_rows']}")

    return overview

def analyze_target_variable(df, target_col='Employed'):
    """
    Analyze the target variable distribution

    Args:
        df (pd.DataFrame): Dataset
        target_col (str): Name of target column

    Returns:
        dict: Target analysis results
    """
    print(f"=== TARGET VARIABLE ANALYSIS: {target_col} ===")

    if target_col not in df.columns:
        print(f"❌ Target column '{target_col}' not found")
        return None

    target_stats = {
        'value_counts': df[target_col].value_counts(),
        'percentages': df[target_col].value_counts(normalize=True) * 100,
        'missing': df[target_col].isnull().sum()
    }

    print("Target Distribution:")
    for value, count in target_stats['value_counts'].items():
        percentage = target_stats['percentages'][value]
        print(f"   {value}: {count} ({percentage:.1f}%)")

    print(f"Missing Values: {target_stats['missing']}")

    # Check for class imbalance
    if len(target_stats['value_counts']) == 2:
        minority_class_pct = target_stats['percentages'].min()
        if minority_class_pct < 10:
            print("⚠️ Severe class imbalance detected (minority class < 10%)")
        elif minority_class_pct < 30:
            print("⚠️ Moderate class imbalance detected (minority class < 30%)")
        else:
            print("✅ Balanced classes")

    return target_stats

def analyze_missing_values(df):
    """
    Comprehensive missing value analysis

    Args:
        df (pd.DataFrame): Dataset

    Returns:
        pd.DataFrame: Missing value summary
    """
    print("=== MISSING VALUES ANALYSIS ===")

    missing_data = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': df.isnull().sum(),
        'Missing_Percentage': (df.isnull().sum() / len(df)) * 100,
        'Data_Type': df.dtypes
    })

    missing_data = missing_data[missing_data['Missing_Count'] > 0].sort_values('Missing_Percentage', ascending=False)

    if len(missing_data) == 0:
        print("✅ No missing values found!")
        return missing_data

    print(f"Columns with missing values: {len(missing_data)}")
    print("\nMissing Value Summary:")
    for _, row in missing_data.head(10).iterrows():
        print(f"   {row['Column']}: {row['Missing_Count']} ({row['Missing_Percentage']:.1f}%)")

    # Categorize severity
    severe = missing_data[missing_data['Missing_Percentage'] > 50]
    moderate = missing_data[(missing_data['Missing_Percentage'] > 20) & (missing_data['Missing_Percentage'] <= 50)]
    mild = missing_data[missing_data['Missing_Percentage'] <= 20]

    print(f"\nSeverity Categories:")
    print(f"   Severe (>50%): {len(severe)} columns")
    print(f"   Moderate (20-50%): {len(moderate)} columns")
    print(f"   Mild (≤20%): {len(mild)} columns")

    return missing_data

def analyze_categorical_variables(df):
    """
    Analyze categorical variables in the dataset

    Args:
        df (pd.DataFrame): Dataset

    Returns:
        dict: Categorical analysis results
    """
    print("=== CATEGORICAL VARIABLES ANALYSIS ===")

    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    if not categorical_cols:
        print("No categorical columns found")
        return {}

    analysis = {}

    for col in categorical_cols:
        unique_count = df[col].nunique()
        most_common = df[col].value_counts().head(3)

        analysis[col] = {
            'unique_count': unique_count,
            'most_common': most_common,
            'missing_count': df[col].isnull().sum()
        }

        print(f"\n{col}:")
        print(f"   Unique values: {unique_count}")
        print(f"   Missing values: {analysis[col]['missing_count']}")
        print(f"   Top 3 values:")
        for value, count in most_common.items():
            percentage = (count / len(df)) * 100
            print(f"      {value}: {count} ({percentage:.1f}%)")

    return analysis

def analyze_numerical_variables(df):
    """
    Analyze numerical variables in the dataset

    Args:
        df (pd.DataFrame): Dataset

    Returns:
        pd.DataFrame: Numerical analysis results
    """
    print("=== NUMERICAL VARIABLES ANALYSIS ===")

    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numerical_cols:
        print("No numerical columns found")
        return pd.DataFrame()

    # Basic statistics
    stats = df[numerical_cols].describe()

    # Additional statistics
    additional_stats = pd.DataFrame({
        'Missing': df[numerical_cols].isnull().sum(),
        'Zeros': (df[numerical_cols] == 0).sum(),
        'Negative': (df[numerical_cols] < 0).sum(),
        'Infinite': np.isinf(df[numerical_cols]).sum()
    })

    # Combine statistics
    comprehensive_stats = pd.concat([stats.T, additional_stats], axis=1)

    print("Numerical Variables Summary:")
    print(comprehensive_stats)

    # Identify potential issues
    print("\nPotential Data Quality Issues:")
    for col in numerical_cols:
        issues = []
        if additional_stats.loc[col, 'Missing'] > 0:
            issues.append(f"Missing: {additional_stats.loc[col, 'Missing']}")
        if additional_stats.loc[col, 'Infinite'] > 0:
            issues.append(f"Infinite: {additional_stats.loc[col, 'Infinite']}")
        if comprehensive_stats.loc[col, 'std'] == 0:
            issues.append("No variation")

        if issues:
            print(f"   {col}: {', '.join(issues)}")

    return comprehensive_stats

def analyze_technology_landscape(df):
    """
    Analyze the technology landscape from HaveWorkedWith column

    Args:
        df (pd.DataFrame): Dataset with HaveWorkedWith column

    Returns:
        dict: Technology analysis results
    """
    print("=== TECHNOLOGY LANDSCAPE ANALYSIS ===")

    if 'HaveWorkedWith' not in df.columns:
        print("❌ HaveWorkedWith column not found")
        return {}

    # Parse all technologies
    all_technologies = []
    tech_per_person = []

    for tech_string in df['HaveWorkedWith'].dropna():
        techs = [tech.strip() for tech in str(tech_string).split(';') if tech.strip()]
        all_technologies.extend(techs)
        tech_per_person.append(len(techs))

    # Technology statistics
    tech_counts = pd.Series(all_technologies).value_counts()

    analysis = {
        'total_unique_technologies': len(tech_counts),
        'most_popular_technologies': tech_counts.head(10),
        'avg_technologies_per_person': np.mean(tech_per_person),
        'median_technologies_per_person': np.median(tech_per_person),
        'technology_distribution': pd.Series(tech_per_person).describe()
    }

    print(f"Total Unique Technologies: {analysis['total_unique_technologies']}")
    print(f"Average Technologies per Person: {analysis['avg_technologies_per_person']:.1f}")
    print(f"Median Technologies per Person: {analysis['median_technologies_per_person']:.1f}")

    print("\nTop 10 Most Popular Technologies:")
    for tech, count in analysis['most_popular_technologies'].items():
        percentage = (count / len(df)) * 100
        print(f"   {tech}: {count} ({percentage:.1f}%)")

    return analysis

def create_eda_visualizations(df, target_col='Employed'):
    """
    Create comprehensive EDA visualizations

    Args:
        df (pd.DataFrame): Dataset
        target_col (str): Target variable column name

    Returns:
        bool: Success status
    """
    if not VISUALIZATION_AVAILABLE:
        print("❌ Visualization libraries not available")
        return False

    print("=== CREATING EDA VISUALIZATIONS ===")

    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")

    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 15))

    # 1. Target variable distribution
    plt.subplot(3, 4, 1)
    if target_col in df.columns:
        df[target_col].value_counts().plot(kind='bar')
        plt.title(f'{target_col} Distribution')
        plt.xlabel(target_col)
        plt.ylabel('Count')

    # 2. Missing values heatmap
    plt.subplot(3, 4, 2)
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0]
    if len(missing_data) > 0:
        missing_data.plot(kind='bar')
        plt.title('Missing Values by Column')
        plt.xticks(rotation=45)
    else:
        plt.text(0.5, 0.5, 'No Missing Values', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Missing Values')

    # 3. Age distribution
    plt.subplot(3, 4, 3)
    if 'Age' in df.columns:
        df['Age'].value_counts().plot(kind='bar')
        plt.title('Age Distribution')
        plt.xlabel('Age Group')

    # 4. Education level
    plt.subplot(3, 4, 4)
    if 'EdLevel' in df.columns:
        df['EdLevel'].value_counts().plot(kind='bar')
        plt.title('Education Level')
        plt.xticks(rotation=45)

    # 5. Years of coding experience
    plt.subplot(3, 4, 5)
    if 'YearsCode' in df.columns:
        df['YearsCode'].hist(bins=20, alpha=0.7)
        plt.title('Years of Coding Experience')
        plt.xlabel('Years')

    # 6. Professional experience
    plt.subplot(3, 4, 6)
    if 'YearsCodePro' in df.columns:
        df['YearsCodePro'].hist(bins=20, alpha=0.7)
        plt.title('Professional Experience')
        plt.xlabel('Years')

    # 7. Computer skills distribution
    plt.subplot(3, 4, 7)
    if 'ComputerSkills' in df.columns:
        df['ComputerSkills'].hist(bins=20, alpha=0.7)
        plt.title('Computer Skills Count')
        plt.xlabel('Number of Skills')

    # 8. Developer type
    plt.subplot(3, 4, 8)
    if 'MainBranch' in df.columns:
        df['MainBranch'].value_counts().plot(kind='bar')
        plt.title('Developer Type')
        plt.xticks(rotation=45)

    # 9. Gender distribution
    plt.subplot(3, 4, 9)
    if 'Gender' in df.columns:
        df['Gender'].value_counts().plot(kind='bar')
        plt.title('Gender Distribution')
        plt.xticks(rotation=45)

    # 10. Mental health
    plt.subplot(3, 4, 10)
    if 'MentalHealth' in df.columns:
        df['MentalHealth'].value_counts().plot(kind='bar')
        plt.title('Mental Health')

    # 11. Employment vs Education (if target available)
    plt.subplot(3, 4, 11)
    if target_col in df.columns and 'EdLevel' in df.columns:
        pd.crosstab(df['EdLevel'], df[target_col]).plot(kind='bar', stacked=True)
        plt.title(f'{target_col} by Education')
        plt.xticks(rotation=45)

    # 12. Salary distribution
    plt.subplot(3, 4, 12)
    if 'PreviousSalary' in df.columns:
        df['PreviousSalary'].hist(bins=20, alpha=0.7)
        plt.title('Salary Distribution')
        plt.xlabel('Salary')

    plt.tight_layout()
    plt.suptitle('Comprehensive EDA Dashboard', fontsize=16, y=1.02)
    plt.show()

    print("✅ EDA visualizations created successfully")
    return True

def generate_eda_report(df, target_col='Employed'):
    """
    Generate a comprehensive EDA report

    Args:
        df (pd.DataFrame): Dataset
        target_col (str): Target variable column name

    Returns:
        dict: Complete EDA report
    """
    print("🚀 GENERATING COMPREHENSIVE EDA REPORT")
    print("=" * 60)

    report = {}

    # Basic overview
    report['overview'] = basic_dataset_overview(df)

    # Target analysis
    report['target_analysis'] = analyze_target_variable(df, target_col)

    # Missing values
    report['missing_values'] = analyze_missing_values(df)

    # Categorical variables
    report['categorical_analysis'] = analyze_categorical_variables(df)

    # Numerical variables
    report['numerical_analysis'] = analyze_numerical_variables(df)

    # Technology landscape
    report['technology_analysis'] = analyze_technology_landscape(df)

    # Create visualizations
    report['visualizations_created'] = create_eda_visualizations(df, target_col)

    print("\n" + "="*60)
    print("🎉 EDA REPORT COMPLETED!")
    print("✅ All analyses completed successfully")
    print("="*60)

    return report

if __name__ == "__main__":
    # Example usage
    print("Exploratory Data Analysis Module")
    print("Use generate_eda_report() for comprehensive analysis")
    print("Individual analysis functions available for targeted exploration")
