"""
Feature Engineering Module for Stack Overflow Developer Survey Analysis

This module contains specialized functions for creating features from the HaveWorkedWith column
and other domain-specific feature engineering tasks.
"""

import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_technology_distribution(df):
    """
    Analyze the distribution of technologies in the dataset
    
    Args:
        df (pd.DataFrame): Dataset with HaveWorkedWith column
        
    Returns:
        pd.Series: Technology frequency counts
    """
    print("=== ANALYZING TECHNOLOGY DISTRIBUTION ===")
    
    # Extract all technologies
    all_technologies = []
    for tech_string in df['HaveWorkedWith'].dropna():
        techs = [tech.strip() for tech in str(tech_string).split(';') if tech.strip()]
        all_technologies.extend(techs)
    
    # Count frequencies
    tech_counts = pd.Series(all_technologies).value_counts()
    
    print(f"Total unique technologies: {len(tech_counts)}")
    print(f"Most common technologies:")
    print(tech_counts.head(10))
    
    return tech_counts

def create_popular_tech_features(df, min_frequency=0.05):
    """
    Create binary features for popular technologies
    
    Args:
        df (pd.DataFrame): Dataset with parsed Technologies_List
        min_frequency (float): Minimum frequency threshold (0.05 = 5%)
        
    Returns:
        pd.DataFrame: Dataset with popular technology binary features
    """
    print(f"=== CREATING POPULAR TECH FEATURES (min_frequency={min_frequency}) ===")
    
    # Get all technologies and their frequencies
    all_technologies = []
    for tech_list in df['Technologies_List']:
        all_technologies.extend(tech_list)
    
    tech_frequency = pd.Series(all_technologies).value_counts()
    
    # Define popularity threshold
    min_count = len(df) * min_frequency
    popular_techs = tech_frequency[tech_frequency >= min_count].index.tolist()
    
    print(f"Found {len(popular_techs)} popular technologies:")
    for tech in popular_techs[:10]:  # Show top 10
        count = tech_frequency[tech]
        percentage = (count / len(df)) * 100
        print(f"   - {tech}: {count} ({percentage:.1f}%)")
    
    # Create binary features for popular technologies
    df_with_popular = df.copy()
    for tech in popular_techs:
        safe_name = tech.replace('/', '_').replace('.', '_').replace(' ', '_').replace('#', 'Sharp').replace('+', 'Plus')
        df_with_popular[f'Tech_{safe_name}'] = df_with_popular['Technologies_List'].apply(
            lambda x: 1 if tech in x else 0
        )
    
    print(f"✅ Created {len(popular_techs)} popular technology features")
    return df_with_popular, popular_techs

def create_technology_stack_features(df):
    """
    Create features for common technology stacks
    
    Args:
        df (pd.DataFrame): Dataset with Technologies_List
        
    Returns:
        pd.DataFrame: Dataset with stack features
    """
    print("=== CREATING TECHNOLOGY STACK FEATURES ===")
    
    # Define common technology stacks
    tech_stacks = {
        'MEAN_Stack': ['MongoDB', 'Express', 'Angular', 'Node.js'],
        'MERN_Stack': ['MongoDB', 'Express', 'React.js', 'Node.js'],
        'MEVN_Stack': ['MongoDB', 'Express', 'Vue.js', 'Node.js'],
        'Django_Stack': ['Python', 'Django', 'PostgreSQL'],
        'Rails_Stack': ['Ruby', 'Ruby on Rails', 'PostgreSQL'],
        'DotNet_Stack': ['C#', 'ASP.NET Core', 'Microsoft SQL Server'],
        'Spring_Stack': ['Java', 'Spring', 'MySQL'],
        'LAMP_Alternative': ['PHP', 'MySQL'],  # Since Linux/Apache not in survey
        'AWS_Stack': ['AWS', 'Docker', 'Git'],
        'Frontend_Modern': ['React.js', 'TypeScript', 'Node.js'],
        'Backend_Python': ['Python', 'Django', 'PostgreSQL'],
        'DevOps_Full': ['Docker', 'Kubernetes', 'AWS', 'Git']
    }
    
    # Full-stack stacks (these indicate complete development capabilities)
    fullstack_stacks = ['MEAN_Stack', 'MERN_Stack', 'MEVN_Stack', 'Django_Stack', 'Rails_Stack', 'DotNet_Stack']
    
    df_with_stacks = df.copy()
    
    # Check for stack completeness
    def check_stack_presence(tech_list, stack_techs):
        matches = sum(1 for tech in stack_techs if tech in tech_list)
        completeness = matches / len(stack_techs) if len(stack_techs) > 0 else 0
        return matches, completeness
    
    # Create stack features
    for stack_name, stack_techs in tech_stacks.items():
        df_with_stacks[f'{stack_name}_Matches'] = df_with_stacks['Technologies_List'].apply(
            lambda x: check_stack_presence(x, stack_techs)[0]
        )
        df_with_stacks[f'{stack_name}_Completeness'] = df_with_stacks['Technologies_List'].apply(
            lambda x: check_stack_presence(x, stack_techs)[1]
        )
        df_with_stacks[f'Has_{stack_name}'] = (df_with_stacks[f'{stack_name}_Completeness'] >= 0.75).astype(int)
    
    # Overall full-stack indicator (enhanced)
    fullstack_columns = [f'Has_{stack}' for stack in fullstack_stacks]
    df_with_stacks['Is_FullStack_Enhanced'] = df_with_stacks[fullstack_columns].max(axis=1)
    
    print(f"✅ Created features for {len(tech_stacks)} technology stacks")
    
    # Show stack adoption rates
    print("Stack adoption rates:")
    for stack in tech_stacks.keys():
        adoption_rate = df_with_stacks[f'Has_{stack}'].mean() * 100
        print(f"   - {stack}: {adoption_rate:.1f}%")
    
    return df_with_stacks

def create_skill_quality_metrics(df):
    """
    Create advanced skill quality and specialization metrics
    
    Args:
        df (pd.DataFrame): Dataset with skill scores
        
    Returns:
        pd.DataFrame: Dataset with quality metrics
    """
    print("=== CREATING SKILL QUALITY METRICS ===")
    
    df_with_quality = df.copy()
    
    # Skill depth vs breadth analysis
    score_columns = ['Programming_Score', 'Web_Score', 'Database_Score', 'CloudDevOps_Score']
    
    # Average skill depth
    df_with_quality['Average_Skill_Depth'] = df_with_quality[score_columns].mean(axis=1)
    
    # Maximum specialization (highest score)
    df_with_quality['Max_Specialization'] = df_with_quality[score_columns].max(axis=1)
    
    # Skill balance (how evenly distributed are the skills)
    df_with_quality['Skill_Balance'] = df_with_quality[score_columns].std(axis=1)
    
    # Primary skill area (which family has the highest score)
    df_with_quality['Primary_Skill_Area'] = df_with_quality[score_columns].idxmax(axis=1)
    df_with_quality['Primary_Skill_Score'] = df_with_quality[score_columns].max(axis=1)
    
    # Specialization vs generalization classification
    df_with_quality['Is_Specialist'] = (
        (df_with_quality['Max_Specialization'] >= 30) & 
        (df_with_quality['Skill_Breadth'] <= 2)
    ).astype(int)
    
    df_with_quality['Is_Generalist'] = (
        (df_with_quality['Skill_Breadth'] >= 3) & 
        (df_with_quality['Average_Skill_Depth'] >= 10)
    ).astype(int)
    
    # Experience-skill alignment
    if 'YearsCode' in df_with_quality.columns:
        df_with_quality['Skill_Learning_Rate'] = df_with_quality['Average_Skill_Depth'] / (df_with_quality['YearsCode'] + 1)
        df_with_quality['Tech_Per_Year'] = df_with_quality['Skill_Breadth'] / (df_with_quality['YearsCode'] + 1)
    
    print("✅ Created advanced skill quality metrics")
    
    return df_with_quality

def create_experience_skill_interactions(df):
    """
    Create interaction features between experience and skills
    
    Args:
        df (pd.DataFrame): Dataset with experience and skill features
        
    Returns:
        pd.DataFrame: Dataset with interaction features
    """
    print("=== CREATING EXPERIENCE-SKILL INTERACTIONS ===")
    
    df_with_interactions = df.copy()
    
    # Basic interaction features
    if all(col in df_with_interactions.columns for col in ['YearsCode', 'Programming_Score']):
        df_with_interactions['Programming_Experience_Interaction'] = (
            df_with_interactions['Programming_Score'] * df_with_interactions['YearsCode']
        )
    
    if all(col in df_with_interactions.columns for col in ['YearsCodePro', 'Is_FullStack']):
        df_with_interactions['Professional_FullStack_Interaction'] = (
            df_with_interactions['YearsCodePro'] * df_with_interactions['Is_FullStack']
        )
    
    # Education-skill interactions
    if all(col in df_with_interactions.columns for col in ['EducationLevel_Numeric', 'Average_Skill_Depth']):
        df_with_interactions['Education_Skill_Interaction'] = (
            df_with_interactions['EducationLevel_Numeric'] * df_with_interactions['Average_Skill_Depth']
        )
    
    # Age-technology adoption
    if all(col in df_with_interactions.columns for col in ['IsYoung', 'CloudDevOps_Score']):
        df_with_interactions['Young_CloudAdoption'] = (
            df_with_interactions['IsYoung'] * df_with_interactions['CloudDevOps_Score']
        )
    
    print("✅ Created experience-skill interaction features")
    
    return df_with_interactions

def analyze_skill_employment_correlation(df, target_col='Employed'):
    """
    Analyze correlation between skills and employment
    
    Args:
        df (pd.DataFrame): Dataset with skill features and employment target
        target_col (str): Name of the employment target column
        
    Returns:
        pd.DataFrame: Correlation analysis results
    """
    print("=== ANALYZING SKILL-EMPLOYMENT CORRELATIONS ===")
    
    if target_col not in df.columns:
        print(f"❌ Target column '{target_col}' not found")
        return None
    
    # Get skill-related columns
    skill_columns = [col for col in df.columns if any(x in col for x in [
        'Score', 'Has_', 'Is_FullStack', 'Skill_', 'Tech_'
    ])]
    
    # Calculate correlations with employment
    correlations = df[skill_columns + [target_col]].corr()[target_col].drop(target_col)
    correlations_sorted = correlations.abs().sort_values(ascending=False)
    
    print(f"Top 10 features correlated with employment:")
    for feature, corr in correlations_sorted.head(10).items():
        direction = "+" if correlations[feature] > 0 else "-"
        print(f"   {direction} {feature}: {abs(corr):.3f}")
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    top_features = correlations_sorted.head(15)
    colors = ['green' if correlations[f] > 0 else 'red' for f in top_features.index]
    
    plt.barh(range(len(top_features)), top_features.values, color=colors, alpha=0.7)
    plt.yticks(range(len(top_features)), top_features.index)
    plt.xlabel('Absolute Correlation with Employment')
    plt.title('Top 15 Features Correlated with Employment Status')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return correlations_sorted

def create_comprehensive_feature_set(df):
    """
    Create a comprehensive set of engineered features
    
    Args:
        df (pd.DataFrame): Dataset with basic preprocessing complete
        
    Returns:
        pd.DataFrame: Dataset with comprehensive feature engineering
    """
    print("🚀 CREATING COMPREHENSIVE FEATURE SET")
    print("=" * 60)
    
    # Step 1: Popular technology features
    df_enhanced, popular_techs = create_popular_tech_features(df)
    
    # Step 2: Technology stack features
    df_enhanced = create_technology_stack_features(df_enhanced)
    
    # Step 3: Skill quality metrics
    df_enhanced = create_skill_quality_metrics(df_enhanced)
    
    # Step 4: Experience-skill interactions
    df_enhanced = create_experience_skill_interactions(df_enhanced)
    
    print("\n" + "="*60)
    print("🎉 COMPREHENSIVE FEATURE ENGINEERING COMPLETED!")
    print(f"✅ Original features: {df.shape[1]}")
    print(f"✅ Enhanced features: {df_enhanced.shape[1]}")
    print(f"✅ New features added: {df_enhanced.shape[1] - df.shape[1]}")
    
    return df_enhanced

if __name__ == "__main__":
    # Example usage - this would typically be called after basic preprocessing
    print("Feature Engineering Module")
    print("This module provides advanced feature engineering functions.")
    print("Use in conjunction with preprocessing.py for complete data preparation.")