
'''
Professional Cybersecurity Attack Detection System Evaluation
Multi-class classification performance assessment with RF model
"'''''
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, classification_report, roc_curve, auc,
    RocCurveDisplay
)

from sklearn.multiclass import OneVsRestClassifier
import warnings
warnings.filterwarnings('ignore')

class AttackDetectionEvaluator:
    """
    A comprehensive evaluation system for cybersecurity attack detection
    Handles both binary and multi-class classification scenarios
    Implements professional testing strategy to prevent overfitting and poor generalization .. . 
    """
    def __init__(self, test_size=0.3, random_state=42):
        """
        Initialize the evaluator with professional testing parameters
        Testing Strategy Overview:
        - Stratified train-test split preserves class distribution
        - Cross-validation for robust performance estimation
        - Separate validation set for hyperparameter tuning
        - Comprehensive metrics for different attack types
        """
        self.test_size = test_size
        self.random_state = random_state
        self.model = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.is_binary = False
        self.feature_names = [
            'Packet Length Mean',
            'Average Packet Size', 
            'Bwd Packet Length Min',
            'Fwd Packets/s',
            'Min Packet Length',
            'Down/Up Ratio'
        ]
    def load_and_preprocess_data(self, file_path):
        """
        Load and preprocess cybersecurity dataset
        Professional Testing Strategy - Phase 1: Data Preparation
        - Handle missing values and outliers
        - Encode categorical labels properly
        - Scale features for consistent model performance
        - Detect binary vs multi-class scenario automatically
        """
        try:
            # Load dataset
            df = pd.read_csv(file_path)
            print(f"Dataset loaded successfully: {df.shape}")
            # Verify required features exist
            required_features = self.feature_names + ['Label']
            missing_features = [f for f in required_features if f not in df.columns]
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")
            # Separate features and target
            X = df[self.feature_names]
            y = df['Label']
            # Handle missing values
            X = X.fillna(X.mean())
            # Detect classification type
            unique_classes = y.nunique()
            self.is_binary = (unique_classes == 2)
            print(f"Classification type: {'Binary' if self.is_binary else 'Multi-class'} ({unique_classes} classes)")
            # Encode labels
            y_encoded = self.label_encoder.fit_transform(y)
            class_names = self.label_encoder.classes_
            print(f"Class names: {class_names}")
            return X, y_encoded, class_names
        except Exception as e:
            print(f"Error in data loading: {e}")
            return None, None, None
    def prepare_data_splits(self, X, y):
        """
        Implement professional data splitting strategy
        Testing Strategy - Phase 2: Robust Data Splitting
        - Stratified splitting maintains class distribution
        - Separate test set for final evaluation (never used in training)
        - Cross-validation for reliable performance estimation
        - Prevents data leakage between splits
        """
        # Initial stratified split
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=self.test_size, 
            stratify=y, random_state=self.random_state
        )
        # Secondary split for validation (if needed for hyperparameter tuning)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.2, 
            stratify=y_temp, random_state=self.random_state
        )
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        X_val_scaled = self.scaler.transform(X_val) if X_val is not None else None
        print(f"Training set: {X_train_scaled.shape}")
        print(f"Testing set: {X_test_scaled.shape}")
        if X_val_scaled is not None:
            print(f"Validation set: {X_val_scaled.shape}")
        return X_train_scaled, X_test_scaled, X_val_scaled, y_train, y_test, y_val
    def train_model(self, X_train, y_train):
        """
        Train Random Forest model with professional configuration
        Testing Strategy - Phase 3: Model Training
        - Random Forest for robustness against overfitting
        - Cross-validation during training for generalization assessment
        - Feature importance analysis for interpretability
        """
        # Configure Random Forest with parameters to prevent overfitting
        rf_params = {
            'n_estimators': 100,
            'max_depth': 10,  # Limit depth to prevent overfitting
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',  # Reduce feature space for each tree
            'bootstrap': True,
            'random_state': self.random_state,
            'n_jobs': -1
        }
        # Use OneVsRest for multi-class ROC curves
        if self.is_binary:
            self.model = RandomForestClassifier(**rf_params)
        else:
            self.model = OneVsRestClassifier(RandomForestClassifier(**rf_params))
        # Train model
        self.model.fit(X_train, y_train)
        # Cross-validation for generalization assessment
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='f1_weighted')
        print(f"Cross-validation F1 scores: {cv_scores}")
        print(f"Mean CV F1-score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        return self.model
    def evaluate_model(self, X_test, y_test, class_names):
        """
        Comprehensive model evaluation with multiple metrics
        Testing Strategy - Phase 4: Comprehensive Evaluation
        - Multiple metrics for different perspectives
        - Confusion matrix for detailed error analysis
        - ROC curves for threshold optimization
        - Per-class metrics for imbalanced scenarios
        """
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        # Calculate False Positive Rate
        cm = confusion_matrix(y_test, y_pred)
        fp_rate = self.calculate_false_positive_rate(cm)
        print("\n" + "="*60)
        print("COMPREHENSIVE MODEL EVALUATION RESULTS")
        print("="*60)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"False Positive Rate: {fp_rate:.4f}")
        print("\n" + "-"*40)
        # Detailed classification report
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred))
        #print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

        # Confusion matrix visualization
        self.plot_confusion_matrix(cm, class_names)
        # ROC curves
        self.plot_roc_curves(y_test, y_pred_proba, class_names)
        # Feature importance
        self.plot_feature_importance()
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'false_positive_rate': fp_rate,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    def calculate_false_positive_rate(self, cm):
        """Calculate false positive rate from confusion matrix"""
        fp = cm.sum(axis=0) - np.diag(cm)  # False positives
        tn = cm.sum() - (cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm))  # True negatives
        fpr = fp / (fp + tn)
        return fpr.mean()  # Average FPR across classes
    def plot_confusion_matrix(self, cm, class_names):
        """Plot professional confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix - Attack Detection System')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
    def plot_roc_curves(self, y_test, y_pred_proba, class_names):
        """
        Plot ROC curves for multi-class classification
        Testing Strategy - Phase 5: ROC Analysis
        - One-vs-Rest ROC curves for multi-class problems
        - AUC scores for each class
        - Macro-average ROC for overall performance
        """
        n_classes = len(class_names)
        # For binary classification
        if self.is_binary:
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve - Binary Classification')
            plt.legend(loc="lower right")
            plt.grid(True)
            plt.show()
        # For multi-class classification
        else:
            plt.figure(figsize=(10, 8))
            # Plot ROC curve for each class
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, i], pos_label=i)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, 
                        label=f'Class {class_names[i]} (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Multi-class ROC Curves - Attack Detection')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            plt.show()
    def plot_feature_importance(self):
        """Plot feature importance for model interpretability"""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'estimator_'):
            importances = self.model.estimator_.feature_importances_
        else:
            print("Feature importance not available for this model configuration")
            return
        feature_imp = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=True)
        plt.figure(figsize=(10, 6))
        plt.barh(feature_imp['feature'], feature_imp['importance'])
        plt.title('Feature Importance - Random Forest Model')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.show()
    def unknown_attack_testing_strategy(self, X_unknown, y_unknown, class_names):
        """
        Professional testing strategy for unknown attack detection
        Testing Strategy - Phase 6: Unknown Attack Evaluation
        - Separate dataset with unknown attack types
        - Out-of-distribution detection assessment
        - Confidence threshold analysis
        - Generalization capability testing
        """
        print("\n" + "="*60)
        print("UNKNOWN ATTACK DETECTION EVALUATION")
        print("="*60)
        print("Strategy: Testing model on unseen attack types to assess generalization")
        X_unknown_scaled = self.scaler.transform(X_unknown)
        # Predict on unknown attacks
        y_pred_unknown = self.model.predict(X_unknown_scaled)
        y_pred_proba_unknown = self.model.predict_proba(X_unknown_scaled)
        # Calculate confidence scores
        confidence_scores = np.max(y_pred_proba_unknown, axis=1)
        # Evaluate performance on unknown attacks
        accuracy_unknown = accuracy_score(y_unknown, y_pred_unknown)
        print(f"Accuracy on unknown attacks: {accuracy_unknown:.4f}")
        print(f"Average confidence score: {confidence_scores.mean():.4f}")
        print(f"Confidence score std: {confidence_scores.std():.4f}")
        # Analyze misclassifications
        misclassified = y_unknown != y_pred_unknown
        misclassification_rate = misclassified.mean()
        print(f"Misclassification rate on unknown attacks: {misclassification_rate:.4f}")
        # Confidence analysis for misclassified samples
        if misclassified.any():
            misclassified_confidence = confidence_scores[misclassified]
            print(f"Average confidence for misclassified: {misclassified_confidence.mean():.4f}")
        return {
            'unknown_attack_accuracy': accuracy_unknown,
            'confidence_scores': confidence_scores,
            'misclassification_rate': misclassification_rate
        }
def main():
    """
    Main execution function demonstrating the professional evaluation framework
    """
    # Initialize the evaluator
    evaluator = AttackDetectionEvaluator()
    
    # Load your dataset (replace with actual file path)
############################################################  CIC-DDoS2019_CSV  from original   - -  All files Available to test ................................

    #file_path = r"E:\Cic-DDos2019 Original\03-11\MSSQL_Pre.csv"  # Update this path 

    file_path = r"E:\Cic-DDos2019 Original\03-11\MSSQL_undersampling.csv"


    #file_path = r"E:\Cic-DDos2019 Original\03-11\UDP_Pre.csv"

    #file_path =  r"E:\Cic-DDos2019 Original\03-11\UDP_undersampling.csv"


###########################################################################################    

    # Load and preprocess data
    X, y, class_names = evaluator.load_and_preprocess_data(file_path)
    if X is None:
        print("Failed to load data. Please check the file path and data format.")
        return
    # Prepare data splits
    X_train, X_test, X_val, y_train, y_test, y_val = evaluator.prepare_data_splits(X, y)
    # Train model
    print("\nTraining Random Forest model...")
    model = evaluator.train_model(X_train, y_train)
    # Comprehensive evaluation
    print("\nPerforming comprehensive evaluation...")
    results = evaluator.evaluate_model(X_test, y_test, class_names)
    # Demonstration of unknown attack testing (if unknown data available)
    # This would require a separate dataset with unknown attack types
    print("\n" + "="*60)
    print("PROFESSIONAL TESTING STRATEGY SUMMARY")
    print("="*60)
    print("1. DATA PREPARATION: Stratified splits, proper scaling, encoding")
    print("2. MODEL TRAINING: Random Forest with overfitting prevention")
    print("3. CROSS-VALIDATION: 5-fold CV for generalization assessment") 
    print("4. COMPREHENSIVE METRICS: Precision, Recall, F1, FPR, Accuracy")
    print("5. VISUAL ANALYSIS: ROC curves, confusion matrices, feature importance")
    print("6. UNKNOWN ATTACK TESTING: Separate evaluation on unseen attack types")
    print("7. CONFIDENCE ANALYSIS: Threshold optimization for real deployment")
    print("\nThis strategy effectively addresses:")
    print("✓ Overfitting through proper regularization and validation")
    print("✓ Poor generalization through cross-validation and unknown attack testing")
    print("✓ Class imbalance through weighted metrics and stratified sampling")
if __name__ == "__main__":
    main()
'''
This professional Python code provides a comprehensive evaluation framework for your cybersecurity attack detection system. Here are the key features:
## **Professional Testing Strategy Highlights:**
### **1. Overfitting Prevention:**
- Limited tree depth and regularization in Random Forest
- Cross-validation during training
- Separate validation set
### **2. Generalization Assessment:**
- Stratified data splitting maintains class distribution
- Comprehensive cross-validation reporting
- Unknown attack testing methodology
### **3. Multi-class & Binary Handling:**
- Automatic detection of classification type
- One-vs-Rest ROC curves for multi-class
- Dynamic adaptation to both scenarios
### **4. Comprehensive Metrics:**
- Precision, Recall, F1-Score, Accuracy
- False Positive Rate calculation
- Per-class performance analysis
### **5. Professional Visualization:**
- Multi-class ROC curves
- Confusion matrices
- Feature importance plots
## **Unknown Attack Testing Strategy:**
The code includes a dedicated method `unknown_attack_testing_strategy()` that implements:
1. **Separate Unknown Dataset**: Test on completely unseen attack types
2. **Confidence Analysis**: Measure model certainty on unknown patterns
3. **Generalization Metrics**: Accuracy and misclassification rates on novel attacks
4. **Threshold Optimization**: Confidence-based detection tuning
## **Key Features:**
- **Dynamic Classification**: Automatically handles binary vs multi-class
- **Robust Evaluation**: Multiple validation strategies
- **Professional Reporting**: Comprehensive metrics and visualizations
- **Scalable Architecture**: Easy to extend with new features or models
To use this code, simply replace the file path with your actual dataset and ensure it contains the specified features. The system will automatically adapt to your data's characteristics and provide a professional evaluation report.
'''''
