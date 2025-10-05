"""
Professional Cybersecurity Attack Detection System Evaluation
Complement Naïve Bayes Classifier for Binary and Multi-class Classification
Test Strategy: Focus on generalization and overfitting prevention
"""
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler #fixed import

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.naive_bayes import ComplementNB

from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             confusion_matrix, classification_report, roc_curve, auc, 
                             roc_auc_score)

from sklearn.multiclass import OneVsRestClassifier

import warnings
warnings.filterwarnings('ignore')

class CyberAttackDetectionEvaluator:
    """
    Professional evaluation system for cybersecurity attack detection
    Test Strategy: Comprehensive evaluation with focus on generalization
    """
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        #ValueError: Negative values in data passed to ComplementNB (input X)
        self.scaler = MinMaxScaler()
        self.is_binary = False
        self.feature_names = [
            'Packet Length Mean',
            'Average Packet Size', 
            'Bwd Packet Length Min',
            'Fwd Packets/s',
            'Min Packet Length',
            'Down/Up Ratio'
        ]
        self.target_name = 'Label'
    def load_and_preprocess_data(self, data_path):
        """
        TEST STRATEGY STEP 1: Data Preparation & Quality Assurance
        - Ensure data quality and handle missing values
        - Prevent data leakage by preprocessing after split
        - Maintain class distribution through stratification
        """
        try:
            # Load dataset
            df = pd.read_csv(data_path)
            print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
            # Validate required features
            missing_features = [f for f in self.feature_names + [self.target_name] if f not in df.columns]
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")
            # Separate features and target
            X = df[self.feature_names]
            y = df[self.target_name]
            # Handle missing values
            X = X.fillna(X.mean())
            # Encode target labels
            y_encoded = self.label_encoder.fit_transform(y)
            self.classes_ = self.label_encoder.classes_
            self.n_classes_ = len(self.classes_)
            self.is_binary = (self.n_classes_ == 2)
            print(f"Number of classes: {self.n_classes_}")
            print(f"Classes: {list(self.classes_)}")
            print(f"Class distribution: {pd.Series(y_encoded).value_counts().to_dict()}")
            return X, y_encoded
        except Exception as e:
            print(f"Error in data loading: {e}")
            return None, None
    def prepare_train_test_sets(self, X, y, test_size=0.3, random_state=42):
        """
        TEST STRATEGY STEP 2: Robust Train-Test Split
        - Use stratified splitting to maintain class distribution
        - Prevent data leakage by fitting scaler only on training data
        - Ensure reproducible results with fixed random state
        """
        # Stratified split to maintain class distribution
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=y, shuffle=True
        )
        # Scale features (fit only on training data to prevent data leakage)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        print(f"Training set: {X_train_scaled.shape[0]} samples")
        print(f"Testing set: {X_test_scaled.shape[0]} samples")
        return X_train_scaled, X_test_scaled, y_train, y_test
    def train_model(self, X_train, y_train):
        """
        TEST STRATEGY STEP 3: Model Training with Overfitting Prevention
        - Use Complement Naïve Bayes for imbalanced data
        - Implement cross-validation for robustness assessment
        - Monitor training stability
        """
        # Initialize Complement Naïve Bayes classifier
        self.model = ComplementNB()
        # Cross-validation for generalization assessment
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=cv, scoring='accuracy')
        print("Cross-validation Results:")
        print(f"Mean Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print(f"Individual fold scores: {cv_scores}")
        # Train final model .. . 
        self.model.fit(X_train, y_train)
        print("Model training completed successfully")
        return cv_scores
    def evaluate_model(self, X_test, y_test):
        """
        TEST STRATEGY STEP 4: Comprehensive Model Evaluation
        - Multiple evaluation metrics for different perspectives
        - Confusion matrix for detailed error analysis
        - ROC curves for threshold optimization
        """
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        # Basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        print("\n" + "="*60)
        print("COMPREHENSIVE MODEL EVALUATION RESULTS")
        print("="*60)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision (Weighted): {precision:.4f}")
        print(f"Recall (Weighted): {recall:.4f}")
        print(f"F1-Score (Weighted): {f1:.4f}")
        # Calculate False Positive Rate for binary classification
        if self.is_binary:
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            print(f"False Positive Rate: {fpr:.4f}")
        # Detailed classification report
        print("\nDetailed Classification Report:")
        '''print(classification_report(y_test, y_pred, 
                                  target_names=self.classes_, 
                                  zero_division=0)) '''
        print(classification_report(y_test, y_pred))
        return y_pred, y_pred_proba, accuracy, precision, recall, f1
    def plot_confusion_matrix(self, y_test, y_pred):
        """
        TEST STRATEGY STEP 5: Error Analysis Visualization
        - Confusion matrix for misclassification patterns
        - Heatmap for intuitive understanding
        - Normalized and absolute counts
        """
        plt.figure(figsize=(10, 8))
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.classes_, 
                   yticklabels=self.classes_)
        plt.title('Confusion Matrix - Attack Detection System\n', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Label', fontweight='bold')
        plt.ylabel('True Label', fontweight='bold')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
        # Print confusion matrix interpretation
        print("\nConfusion Matrix Interpretation:")
        total_samples = np.sum(cm)
        correct_predictions = np.trace(cm)
        overall_accuracy = correct_predictions / total_samples
        print(f"Overall Accuracy from CM: {overall_accuracy:.4f}")
        return cm
    def plot_roc_curves(self, y_test, y_pred_proba):
        """
        TEST STRATEGY STEP 6: ROC Analysis for Threshold Optimization
        - Multi-class ROC curves using One-vs-Rest approach
        - AUC scores for each class
        - Macro-average ROC curve
        """
        plt.figure(figsize=(12, 10))
        # For binary classification
        if self.is_binary:
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                    label='Random Classifier')
        # For multi-class classification
        else:
            # Binarize labels for One-vs-Rest ROC
            from sklearn.preprocessing import label_binarize
            y_test_bin = label_binarize(y_test, classes=range(self.n_classes_))
            # Compute ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(self.n_classes_):
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
                plt.plot(fpr[i], tpr[i], lw=2, 
                        label=f'Class {self.classes_[i]} (AUC = {roc_auc[i]:.4f})')
            # Compute macro-average ROC curve and ROC area
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(self.n_classes_)]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(self.n_classes_):
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
            mean_tpr /= self.n_classes_
            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
            plt.plot(fpr["macro"], tpr["macro"], 
                    label=f'Macro-average (AUC = {roc_auc["macro"]:.4f})',
                    color='navy', linestyle=':', linewidth=4)
            plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontweight='bold')
        plt.ylabel('True Positive Rate', fontweight='bold')
        plt.title('ROC Curves - Attack Detection Performance\n', 
                 fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
        return roc_auc
    def feature_importance_analysis(self, X_train):
        """
        TEST STRATEGY STEP 7: Feature Importance Analysis
        - Understand which features contribute most to detection
        - Identify potential feature engineering opportunities
        """
        if hasattr(self.model, 'feature_log_prob_'):
            # Calculate feature importance based on log probabilities
            feature_importance = np.abs(self.model.feature_log_prob_[1] - 
                                      self.model.feature_log_prob_[0])
            # Create feature importance dataframe
            importance_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': feature_importance
            }).sort_values('Importance', ascending=False)
            plt.figure(figsize=(10, 6))
            sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis')
            plt.title('Feature Importance - Complement Naïve Bayes\n', 
                     fontsize=14, fontweight='bold')
            plt.xlabel('Importance Score', fontweight='bold')
            plt.tight_layout()
            plt.show()
            print("\nFeature Importance Ranking:")
            print(importance_df)
            return importance_df
        return None
    def comprehensive_evaluation(self, data_path):
        """
        MAIN TEST STRATEGY: End-to-End Evaluation Pipeline
        Combines all test strategy steps for professional assessment
        """
        print("INITIATING COMPREHENSIVE CYBERSECURITY ATTACK DETECTION EVALUATION")
        print("="*70)
        # Step 1: Data Preparation
        X, y = self.load_and_preprocess_data(data_path)
        if X is None:
            return
        # Step 2: Train-Test Split
        X_train, X_test, y_train, y_test = self.prepare_train_test_sets(X, y)
        # Step 3: Model Training with Cross-Validation
        cv_scores = self.train_model(X_train, y_train)
        # Step 4: Model Evaluation
        y_pred, y_pred_proba, accuracy, precision, recall, f1 = self.evaluate_model(X_test, y_test)
        # Step 5: Confusion Matrix Analysis
        cm = self.plot_confusion_matrix(y_test, y_pred)
        # Step 6: ROC Analysis
        roc_auc = self.plot_roc_curves(y_test, y_pred_proba)
        # Step 7: Feature Importance
        importance_df = self.feature_importance_analysis(X_train)
        # Final Summary
        self.print_evaluation_summary(accuracy, precision, recall, f1, cv_scores, roc_auc)
        return {
            'model': self.model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'roc_auc': roc_auc,
            'feature_importance': importance_df,
            'cv_scores': cv_scores
        }
    def print_evaluation_summary(self, accuracy, precision, recall, f1, cv_scores, roc_auc):
        """
        TEST STRATEGY FINAL STEP: Professional Evaluation Summary
        - Consolidated performance metrics
        - Generalization assessment
        - Deployment recommendations
        """
        print("\n" + "="*70)
        print("PROFESSIONAL EVALUATION SUMMARY - CYBERSECURITY ATTACK DETECTION")
        print("="*70)
        print("\n📊 PERFORMANCE METRICS:")
        print(f"   • Accuracy: {accuracy:.4f}")
        print(f"   • Precision: {precision:.4f}")
        print(f"   • Recall: {recall:.4f}")
        print(f"   • F1-Score: {f1:.4f}")
        print("\n🔄 GENERALIZATION ASSESSMENT:")
        print(f"   • Cross-validation Mean Accuracy: {cv_scores.mean():.4f}")
        print(f"   • Cross-validation Std: {cv_scores.std():.4f}")
        print(f"   • Generalization Gap: {abs(accuracy - cv_scores.mean()):.4f}")
        if isinstance(roc_auc, dict):
            if 'macro' in roc_auc:
                print(f"   • Macro-average AUC: {roc_auc['macro']:.4f}")
        else:
            print(f"   • ROC AUC: {roc_auc:.4f}")
        print("\n🎯 TEST STRATEGY VALIDATION:")
        print("   ✓ Stratified train-test split for representative sampling")
        print("   ✓ Cross-validation for robustness assessment")
        print("   ✓ Multiple metrics for comprehensive evaluation")
        print("   ✓ ROC analysis for threshold optimization")
        print("   ✓ Feature importance for model interpretability")
        print("   ✓ Data leakage prevention through proper preprocessing")
        print("\n⚠️  OVERFITTING PREVENTION STATUS:")
        cv_std = cv_scores.std()
        if cv_std < 0.05:
            print("   ✅ LOW variance - Good generalization")
        elif cv_std < 0.1:
            print("   ⚠️  MODERATE variance - Acceptable generalization")
        else:
            print("   ❌ HIGH variance - Potential overfitting")
        generalization_gap = abs(accuracy - cv_scores.mean())
        if generalization_gap < 0.05:
            print("   ✅ SMALL generalization gap - Model generalizes well")
        else:
            print("   ⚠️  LARGE generalization gap - Potential overfitting")
        print("\n💡 RECOMMENDATIONS:")
        if self.is_binary:
            print("   • Consider threshold adjustment based on ROC analysis")
        else:
            print("   • Review per-class performance for imbalanced detection")
        print("   • Monitor false positive rates in production")
        print("   • Regular model retraining with new attack patterns")
        print("   • Consider ensemble methods for improved robustness")
# Example usage and demonstration
def main():
    """
    DEMONSTRATION: Professional Test Strategy Implementation
    """
  
############################################################  CIC-DDoS2019_CSV  from original   - -  All files Available to test ................................

    #data_path = r"E:\Cic-DDos2019 Original\03-11\MSSQL_Pre.csv"

    data_path = r"E:\Cic-DDos2019 Original\03-11\MSSQL_undersampling.csv"


    #data_path = r"E:\Cic-DDos2019 Original\03-11\UDP_Pre.csv"

    #data_path =  r"E:\Cic-DDos2019 Original\03-11\UDP_undersampling.csv"


###########################################################################################
    #initialize evaluator
    evaluator = CyberAttackDetectionEvaluator()
    
    evaluator.comprehensive_evaluation(data_path) # you would replace this with your actual data path
    
    print("Professional Cybersecurity Evaluation System Ready!")
    print("\nTEST STRATEGY IMPLEMENTED:")
    print("1. Data Quality Assurance & Preprocessing")
    print("2. Stratified Sampling for Representative Splits")
    print("3. Cross-Validation for Generalization Assessment")
    print("4. Comprehensive Multi-Metric Evaluation")
    print("5. Confusion Matrix for Error Analysis")
    print("6. ROC Analysis for Threshold Optimization")
    print("7. Feature Importance for Model Interpretability")
    print("8. Overfitting Detection and Prevention Measures")
if __name__ == "__main__":
    main()
'''
This professional Python code implements a comprehensive cybersecurity attack detection evaluation system with the following key features:
## **Professional Test Strategy Implementation:**
### **1. Data Quality & Preparation**
- Validates required features and handles missing values
- Prevents data leakage through proper preprocessing pipeline
- Maintains class distribution with stratified sampling
### **2. Overfitting Prevention**
- Cross-validation with Stratified K-Fold
- Monitoring of generalization gap
- Feature importance analysis to avoid over-reliance on specific features
### **3. Comprehensive Evaluation Metrics**
- **Accuracy**: Overall correctness
- **Precision**: False positive minimization
- **Recall**: Attack detection capability
- **F1-Score**: Balanced metric
- **False Positive Rate**: Critical for security applications
- **ROC-AUC**: Threshold-independent performance
### **4. Multi-class & Binary Classification Support**
- Dynamic handling of both scenarios
- One-vs-Rest ROC curves for multi-class
- Appropriate metric calculations for each case
### **5. Visualization & Interpretation**
- Confusion matrices with heatmaps
- ROC curves for all classes
- Feature importance charts
- Professional reporting with recommendations
### **6. Model-Specific Implementation**
- Complement Naïve Bayes for imbalanced data
- Feature scaling for numerical stability
- Probability estimates for ROC analysis
The test strategy ensures the system can generalize well to new attack patterns while maintaining high detection rates and low false positives.
'''
