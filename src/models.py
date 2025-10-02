"""
Machine Learning Models Module for Employment Prediction

This module contains various machine learning models and evaluation functions
for predicting employment status based on developer skills and experience.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, precision_recall_curve
)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
import joblib
import os

class EmploymentPredictor:
    """
    Main class for employment prediction with multiple ML algorithms
    """
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_score = 0
        self.results = {}
        
    def initialize_models(self):
        """Initialize all machine learning models"""
        print("=== INITIALIZING ML MODELS ===")
        
        self.models = {
            'Logistic_Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random_Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient_Boosting': GradientBoostingClassifier(random_state=42),
            'SVM': SVC(random_state=42, probability=True),
            'Naive_Bayes': GaussianNB(),
            'KNN': KNeighborsClassifier(n_neighbors=5)
        }
        
        print(f"✅ Initialized {len(self.models)} models:")
        for name in self.models.keys():
            print(f"   - {name}")
    
    def train_single_model(self, name, model, X_train, y_train, X_test, y_test):
        """
        Train and evaluate a single model
        
        Args:
            name (str): Model name
            model: ML model instance
            X_train, y_train: Training data
            X_test, y_test: Test data
            
        Returns:
            dict: Model performance metrics
        """
        print(f"Training {name}...")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'roc_auc': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
        }
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        metrics['cv_mean'] = cv_scores.mean()
        metrics['cv_std'] = cv_scores.std()
        
        # Store predictions for later analysis
        metrics['predictions'] = y_pred
        metrics['probabilities'] = y_pred_proba
        
        print(f"   Accuracy: {metrics['accuracy']:.3f}")
        print(f"   F1-Score: {metrics['f1_score']:.3f}")
        print(f"   CV Score: {metrics['cv_mean']:.3f} ± {metrics['cv_std']:.3f}")
        
        return metrics
    
    def train_all_models(self, X_train, y_train, X_test, y_test):
        """
        Train all models and compare performance
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            
        Returns:
            dict: All model results
        """
        print("🚀 TRAINING ALL MODELS")
        print("=" * 50)
        
        if not self.models:
            self.initialize_models()
        
        self.results = {}
        
        for name, model in self.models.items():
            try:
                metrics = self.train_single_model(name, model, X_train, y_train, X_test, y_test)
                self.results[name] = metrics
                
                # Track best model
                if metrics['f1_score'] > self.best_score:
                    self.best_score = metrics['f1_score']
                    self.best_model = name
                    
            except Exception as e:
                print(f"❌ Error training {name}: {e}")
                continue
        
        print("\n" + "=" * 50)
        print("🎉 MODEL TRAINING COMPLETED!")
        print(f"✅ Best Model: {self.best_model} (F1: {self.best_score:.3f})")
        
        return self.results
    
    def hyperparameter_tuning(self, X_train, y_train, model_name=None):
        """
        Perform hyperparameter tuning for specified model or best model
        
        Args:
            X_train, y_train: Training data
            model_name (str): Model to tune (None for best model)
            
        Returns:
            dict: Best parameters and score
        """
        target_model = model_name or self.best_model
        
        if not target_model or target_model not in self.models:
            print("❌ No valid model specified for tuning")
            return None
        
        print(f"=== HYPERPARAMETER TUNING: {target_model} ===")
        
        # Define parameter grids
        param_grids = {
            'Logistic_Regression': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            },
            'Random_Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5, 10]
            },
            'Gradient_Boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'SVM': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            },
            'KNN': {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        }
        
        if target_model not in param_grids:
            print(f"❌ No parameter grid defined for {target_model}")
            return None
        
        # Perform grid search
        model = self.models[target_model]
        param_grid = param_grids[target_model]
        
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1, verbose=1
        )
        
        print("Starting grid search...")
        grid_search.fit(X_train, y_train)
        
        tuning_results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_estimator': grid_search.best_estimator_
        }
        
        print(f"✅ Best Parameters: {tuning_results['best_params']}")
        print(f"✅ Best CV Score: {tuning_results['best_score']:.3f}")
        
        # Update the model with best parameters
        self.models[target_model] = grid_search.best_estimator_
        
        return tuning_results
    
    def generate_model_comparison(self):
        """
        Generate a comprehensive model comparison report
        
        Returns:
            pd.DataFrame: Model comparison results
        """
        print("=== MODEL COMPARISON REPORT ===")
        
        if not self.results:
            print("❌ No model results available. Train models first.")
            return None
        
        # Create comparison dataframe
        comparison_data = []
        for model_name, metrics in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1_Score': metrics['f1_score'],
                'ROC_AUC': metrics['roc_auc'] if metrics['roc_auc'] else 'N/A',
                'CV_Mean': metrics['cv_mean'],
                'CV_Std': metrics['cv_std']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('F1_Score', ascending=False)
        
        print("\nModel Performance Comparison:")
        print(comparison_df.round(3))
        
        # Highlight best model
        best_model_row = comparison_df.iloc[0]
        print(f"\n🏆 BEST MODEL: {best_model_row['Model']}")
        print(f"   F1-Score: {best_model_row['F1_Score']:.3f}")
        print(f"   Accuracy: {best_model_row['Accuracy']:.3f}")
        print(f"   ROC-AUC: {best_model_row['ROC_AUC']}")
        
        return comparison_df
    
    def create_evaluation_plots(self, X_test, y_test):
        """
        Create comprehensive evaluation plots
        
        Args:
            X_test, y_test: Test data
            
        Returns:
            bool: Success status
        """
        if not VISUALIZATION_AVAILABLE:
            print("❌ Visualization libraries not available")
            return False
        
        if not self.results:
            print("❌ No model results available")
            return False
        
        print("=== CREATING EVALUATION PLOTS ===")
        
        # Set up the plotting
        n_models = len(self.results)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model Evaluation Dashboard', fontsize=16)
        
        # 1. Model Comparison Bar Chart
        ax1 = axes[0, 0]
        models = list(self.results.keys())
        f1_scores = [self.results[model]['f1_score'] for model in models]
        
        bars = ax1.bar(models, f1_scores, color='skyblue', alpha=0.7)
        ax1.set_title('F1-Score Comparison')
        ax1.set_ylabel('F1-Score')
        ax1.tick_params(axis='x', rotation=45)
        
        # Highlight best model
        best_idx = f1_scores.index(max(f1_scores))
        bars[best_idx].set_color('gold')
        
        # 2. ROC Curves
        ax2 = axes[0, 1]
        for model_name, metrics in self.results.items():
            if metrics['probabilities'] is not None:
                fpr, tpr, _ = roc_curve(y_test, metrics['probabilities'])
                auc_score = metrics['roc_auc']
                ax2.plot(fpr, tpr, label=f'{model_name} (AUC: {auc_score:.3f})')
        
        ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curves')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Confusion Matrix for Best Model
        ax3 = axes[0, 2]
        if self.best_model and self.best_model in self.results:
            best_predictions = self.results[self.best_model]['predictions']
            cm = confusion_matrix(y_test, best_predictions)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3)
            ax3.set_title(f'Confusion Matrix - {self.best_model}')
            ax3.set_xlabel('Predicted')
            ax3.set_ylabel('Actual')
        
        # 4. Cross-Validation Scores
        ax4 = axes[1, 0]
        cv_means = [self.results[model]['cv_mean'] for model in models]
        cv_stds = [self.results[model]['cv_std'] for model in models]
        
        ax4.bar(models, cv_means, yerr=cv_stds, capsize=5, color='lightcoral', alpha=0.7)
        ax4.set_title('Cross-Validation Scores')
        ax4.set_ylabel('CV Accuracy')
        ax4.tick_params(axis='x', rotation=45)
        
        # 5. Precision vs Recall
        ax5 = axes[1, 1]
        precisions = [self.results[model]['precision'] for model in models]
        recalls = [self.results[model]['recall'] for model in models]
        
        scatter = ax5.scatter(recalls, precisions, c=f1_scores, cmap='viridis', s=100, alpha=0.7)
        for i, model in enumerate(models):
            ax5.annotate(model, (recalls[i], precisions[i]), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8)
        
        ax5.set_xlabel('Recall')
        ax5.set_ylabel('Precision')
        ax5.set_title('Precision vs Recall')
        plt.colorbar(scatter, ax=ax5, label='F1-Score')
        
        # 6. Feature Importance (for tree-based models)
        ax6 = axes[1, 2]
        if self.best_model in ['Random_Forest', 'Gradient_Boosting']:
            model = self.models[self.best_model]
            if hasattr(model, 'feature_importances_'):
                # Assuming we have feature names available
                # This would need to be passed in or stored separately
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1][:10]  # Top 10
                
                ax6.bar(range(10), importances[indices])
                ax6.set_title(f'Top 10 Feature Importances - {self.best_model}')
                ax6.set_xlabel('Features')
                ax6.set_ylabel('Importance')
            else:
                ax6.text(0.5, 0.5, 'Feature importance not available', 
                        ha='center', va='center', transform=ax6.transAxes)
                ax6.set_title('Feature Importance')
        else:
            ax6.text(0.5, 0.5, f'{self.best_model} does not support feature importance', 
                    ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('Feature Importance')
        
        plt.tight_layout()
        plt.show()
        
        print("✅ Evaluation plots created successfully")
        return True
    
    def save_best_model(self, filepath):
        """
        Save the best performing model
        
        Args:
            filepath (str): Path to save the model
            
        Returns:
            bool: Success status
        """
        if not self.best_model or self.best_model not in self.models:
            print("❌ No best model available to save")
            return False
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save the model
            joblib.dump(self.models[self.best_model], filepath)
            
            # Save model metadata
            metadata = {
                'model_name': self.best_model,
                'performance': self.results[self.best_model],
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            metadata_path = filepath.replace('.pkl', '_metadata.json')
            pd.Series(metadata).to_json(metadata_path)
            
            print(f"✅ Best model saved to: {filepath}")
            print(f"✅ Metadata saved to: {metadata_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return False
    
    def load_model(self, filepath):
        """
        Load a saved model
        
        Args:
            filepath (str): Path to the saved model
            
        Returns:
            bool: Success status
        """
        try:
            model = joblib.load(filepath)
            
            # Try to load metadata
            metadata_path = filepath.replace('.pkl', '_metadata.json')
            if os.path.exists(metadata_path):
                metadata = pd.read_json(metadata_path, typ='series')
                model_name = metadata.get('model_name', 'Loaded_Model')
                print(f"✅ Model loaded: {model_name}")
            else:
                model_name = 'Loaded_Model'
                print("✅ Model loaded (no metadata found)")
            
            self.models[model_name] = model
            self.best_model = model_name
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

def quick_employment_prediction(X_train, y_train, X_test, y_test):
    """
    Quick function to train and evaluate multiple models
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        
    Returns:
        EmploymentPredictor: Trained predictor instance
    """
    print("🚀 QUICK EMPLOYMENT PREDICTION")
    print("=" * 50)
    
    # Initialize predictor
    predictor = EmploymentPredictor()
    
    # Train all models
    results = predictor.train_all_models(X_train, y_train, X_test, y_test)
    
    # Generate comparison
    comparison = predictor.generate_model_comparison()
    
    # Create evaluation plots
    predictor.create_evaluation_plots(X_test, y_test)
    
    return predictor

if __name__ == "__main__":
    print("Machine Learning Models Module")
    print("Use EmploymentPredictor class for comprehensive model training and evaluation")
    print("Use quick_employment_prediction() for rapid prototyping")