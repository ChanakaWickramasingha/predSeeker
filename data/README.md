# Data Directory

## Structure

### raw/
Place your original dataset files here:
- `stackoverflow_with_nulls.csv` - The main survey dataset

### processed/
Processed datasets will be saved here automatically:
- `preprocessed_data.csv` - After running preprocessing
- `feature_engineered_data.csv` - After feature engineering
- `train_test_split.csv` - Final train/test datasets

## Dataset Information

### Main Dataset: stackoverflow_with_nulls.csv
Expected columns:
- `Employed` - Target variable (1=Employed, 0=Not Employed)
- `HaveWorkedWith` - Semicolon-separated technology list
- `ComputerSkills` - Number of computer skills
- `YearsCode` - Years of coding experience
- `YearsCodePro` - Years of professional coding
- `Age` - Age group (<35 or >35)
- `EdLevel` - Education level
- `Gender` - Gender identity
- `MentalHealth` - Mental health considerations
- `MainBranch` - Developer type
- `PreviousSalary` - Previous salary information
- Other demographic and experience columns

### Processing Notes
- Missing values will be handled during preprocessing
- Technology strings will be parsed into skill families
- New features will be engineered from existing columns