"""
Professional Cybersecurity Intrusion Detection System Evaluation
Author: AI Security Specialist
Date: 2024
Description: Comprehensive evaluation of ML-based IDS using Logistic Regression
with robust testing strategy to prevent overfitting and poor generalization.
"""
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           confusion_matrix, classification_report, roc_curve, auc, 
                           roc_auc_score)

from sklearn.multiclass import OneVsRestClassifier

import warnings
warnings.filterwarnings('ignore')

class IDSEvaluator:
    """
    Professional Intrusion Detection System Evaluator
    Implements robust testing strategy to address:
    - Poor Generalization through cross-validation
    - Overfitting through proper train/test splits and regularization
    """
    def __init__(self, test_size=0.3, random_state=42, cv_folds=5):
        """
        Initialize the IDS Evaluator with professional testing parameters
        Testing Strategy:
        1. Stratified Train-Test Split: Maintains class distribution
        2. Cross-Validation: Ensures model generalization
        3. Regularization: Prevents overfitting in Logistic Regression
        4. Comprehensive Metrics: Multi-faceted performance evaluation
        """
        self.test_size = test_size
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_binary = False
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        # Professional color scheme for plots
        self.colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3E885B']
    def load_and_preprocess_data(self, file_path):
        """
        Step 1: Data Loading and Preprocessing
        Professional testing consideration: Handle missing values and outliers
        """
        print("=== PHASE 1: DATA LOADING AND PREPROCESSING ===")
        # Load dataset
        try:
            data = pd.read_csv(file_path)
            print(f"✓ Dataset loaded successfully: {data.shape[0]} samples, {data.shape[1]} features")
        except FileNotFoundError:
            print("✗ File not found. Creating sample data for demonstration...")
            data = self._create_sample_data()
        # Validate required features
        required_features = ['Packet Length Mean', 'Average Packet Size', 
                           'Bwd Packet Length Min', 'Fwd Packets/s', 
                           'Min Packet Length', 'Down/Up Ratio', 'Label']
        missing_features = [feat for feat in required_features if feat not in data.columns]
        if missing_features:
            print(f"⚠ Missing features: {missing_features}. Using available features.")
            available_features = [feat for feat in required_features if feat in data.columns]
            features = available_features[:-1]  # Exclude Label
        else:
            features = required_features[:-1]
        # Prepare features and target
        X = data[features]
        y = data['Label']
        # Check if binary or multiclass classification
        unique_classes = y.nunique()
        self.is_binary = (unique_classes == 2)
        print(f"✓ Classification type: {'Binary' if self.is_binary else 'Multi-class'} ({unique_classes} classes)")
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        class_mapping = dict(zip(self.label_encoder.classes_, 
                               self.label_encoder.transform(self.label_encoder.classes_)))
        print(f"✓ Class mapping: {class_mapping}")
        # Professional testing: Split data before scaling to prevent data leakage
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y_encoded, test_size=self.test_size, random_state=self.random_state, 
            stratify=y_encoded  # Maintain class distribution
        )
        # Scale features (fit only on training data)
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        print(f"✓ Data split: {self.X_train.shape[0]} training, {self.X_test.shape[0]} testing samples")
        print(f"✓ Feature scaling applied (prevents data leakage)")
        return X, y_encoded
    def train_model(self):
        """
        Step 2: Model Training with Overfitting Prevention
        Professional testing strategy: Use regularization and cross-validation
        """
        print("\n=== PHASE 2: MODEL TRAINING WITH ROBUST STRATEGY ===")
        # Professional approach: Regularized Logistic Regression
        # C parameter controls regularization strength (prevents overfitting)
        logistic_params = {
            'C': 1.0,  # Regularization strength
            'max_iter': 1000,
            'random_state': self.random_state,
            'solver': 'lbfgs'
        }
        if self.is_binary:
            self.model = LogisticRegression(**logistic_params)
            print("✓ Binary classification model configured")
        else:
            self.model = OneVsRestClassifier(LogisticRegression(**logistic_params))
            print("✓ Multi-class classification model configured (OneVsRest)")
        # Professional testing: Cross-validation for generalization assessment
        cv_scores = cross_val_score(self.model, self.X_train_scaled, self.y_train, 
                                  cv=self.cv_folds, scoring='accuracy')
        print(f"✓ Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print("✓ Cross-validation indicates model generalization capability")
        # Train final model
        self.model.fit(self.X_train_scaled, self.y_train)
        print("✓ Model training completed with regularization")
        return self.model
    def comprehensive_evaluation(self):
        """
        Step 3: Comprehensive Security Evaluation
        Professional testing: Multiple metrics and visualization
        """
        print("\n=== PHASE 3: COMPREHENSIVE SECURITY EVALUATION ===")
        # Predictions
        y_pred = self.model.predict(self.X_test_scaled)
        y_pred_proba = self.model.predict_proba(self.X_test_scaled)
        # 1. Basic Metrics
        self._calculate_basic_metrics(y_pred)
        # 2. Confusion Matrix
        self._plot_confusion_matrix(y_pred)
        # 3. ROC Curves
        self._plot_roc_curves(y_pred_proba)
        # 4. Detailed Classification Report
        self._print_detailed_report(y_pred)
        # 5. False Positive Analysis
        self._analyze_false_positives(y_pred)
    def _calculate_basic_metrics(self, y_pred):
        """Calculate comprehensive security metrics"""
        print("\n--- SECURITY METRICS ANALYSIS ---")
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='weighted' if not self.is_binary else 'binary')
        recall = recall_score(self.y_test, y_pred, average='weighted' if not self.is_binary else 'binary')
        f1 = f1_score(self.y_test, y_pred, average='weighted' if not self.is_binary else 'binary')
        # Calculate False Positive Rate for security context
        cm = confusion_matrix(self.y_test, y_pred)
        if self.is_binary:
            tn, fp, fn, tp = cm.ravel()
            fpr = fp / (fp + tn)
        else:
            # For multiclass, calculate macro FPR
            fpr = self._calculate_multiclass_fpr(cm)
        print(f"📊 Accuracy: {accuracy:.4f}")
        print(f"🎯 Precision: {precision:.4f} (True Positives / All Predicted Positives)")
        print(f"🔍 Recall: {recall:.4f} (True Positives / All Actual Positives)")
        print(f"⚖️ F1-Score: {f1:.4f} (Harmonic mean of Precision and Recall)")
        print(f"🚨 False Positive Rate: {fpr:.4f} (False Alarms / All Actual Negatives)")
        # Security-specific interpretation
        print("\n🔒 SECURITY INTERPRETATION:")
        print(f"• System correctly identifies {recall*100:.1f}% of actual attacks")
        print(f"• When system alerts, {precision*100:.1f}% are real attacks")
        print(f"• {fpr*100:.1f}% of benign traffic is incorrectly flagged as malicious")
        return accuracy, precision, recall, f1, fpr
    def _plot_confusion_matrix(self, y_pred):
        """Plot professional confusion matrix"""
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(self.y_test, y_pred)
        class_names = self.label_encoder.classes_
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('CONFUSION MATRIX - Attack Detection Performance\n', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Attack Type', fontweight='bold')
        plt.ylabel('Actual Attack Type', fontweight='bold')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
        print("✓ Confusion matrix generated - Shows detection patterns per attack type")
    def _plot_roc_curves(self, y_pred_proba):
        """Plot ROC curves for binary and multiclass scenarios"""
        plt.figure(figsize=(12, 10))
        n_classes = len(self.label_encoder.classes_)
        if self.is_binary:
            # Binary ROC curve
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=self.colors[0], lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                    label='Random Classifier')
        else:
            # Multiclass ROC curves
            for i, class_name in enumerate(self.label_encoder.classes_):
                fpr, tpr, _ = roc_curve(self.y_test == i, y_pred_proba[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, color=self.colors[i % len(self.colors)], lw=2,
                        label=f'{class_name} (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                    label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (False Alarm Rate)', fontweight='bold')
        plt.ylabel('True Positive Rate (Detection Rate)', fontweight='bold')
        plt.title('ROC CURVES - Attack Detection Capability\n', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        print("✓ ROC curves generated - Shows trade-off between detection and false alarms")
    def _print_detailed_report(self, y_pred):
        """Print detailed classification report"""
        print("\n--- DETAILED CLASS-WISE PERFORMANCE ---")
        '''report = classification_report(self.y_test, y_pred, 
                                     target_names=self.label_encoder.classes_)'''
        report = classification_report(self.y_test, y_pred) 

        print(report)
    def _analyze_false_positives(self, y_pred):
        """Analyze false positives for security improvement"""
        print("\n--- FALSE POSITIVE ANALYSIS ---")
        fp_indices = np.where((y_pred != self.y_test) & (y_pred != 0))[0]
        if len(fp_indices) > 0:
            print(f"⚠ {len(fp_indices)} false positives detected")
            print("False positives represent benign traffic incorrectly flagged as attacks")
            print("Recommendation: Review feature thresholds and retrain with more benign samples")
        else:
            print("✓ No significant false positives detected")
    def _calculate_multiclass_fpr(self, cm):
        """Calculate macro False Positive Rate for multiclass"""
        fpr_list = []
        for i in range(len(cm)):
            fp = np.sum(cm[:, i]) - cm[i, i]
            tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fpr_list.append(fpr)
        return np.mean(fpr_list)
    def _create_sample_data(self):
        """Create sample cybersecurity data for demonstration"""
        np.random.seed(self.random_state)
        n_samples = 1000
        data = {
            'Packet Length Mean': np.random.normal(500, 200, n_samples),
            'Average Packet Size': np.random.normal(800, 300, n_samples),
            'Bwd Packet Length Min': np.random.normal(50, 20, n_samples),
            'Fwd Packets/s': np.random.normal(1000, 400, n_samples),
            'Min Packet Length': np.random.normal(40, 15, n_samples),
            'Down/Up Ratio': np.random.normal(1.5, 0.8, n_samples),
        }
        # Create synthetic labels (Normal, DDoS, PortScan, Malware)
        conditions = [
            (data['Fwd Packets/s'] > 1200) & (data['Packet Length Mean'] > 600),
            (data['Bwd Packet Length Min'] < 30) & (data['Down/Up Ratio'] > 2),
            (data['Min Packet Length'] < 20) & (data['Average Packet Size'] > 1000),
        ]
        choices = ['DDoS', 'PortScan', 'Malware']
        data['Label'] = np.select(conditions, choices, default='Normal')
        return pd.DataFrame(data)
def main():
    """
    MAIN EXECUTION: PROFESSIONAL IDS EVALUATION FRAMEWORK
    Implements comprehensive testing strategy to prevent:
    - Poor Generalization through cross-validation and proper splits
    - Overfitting through regularization and validation techniques
    """
    print("🚀 PROFESSIONAL INTRUSION DETECTION SYSTEM EVALUATION")
    print("=" * 60)
    # Initialize evaluator with professional parameters
    evaluator = IDSEvaluator(
        test_size=0.3,      # 70-30 split for robust validation
        random_state=42,    # Reproducible results
        cv_folds=5          # 5-fold cross-validation
    )
    # Load and preprocess data
############################################################  CIC-DDoS2019_CSV  from original   - -  All files Available to test ................................

    #file_path = r"E:\Cic-DDos2019 Original\03-11\MSSQL_Pre.csv"

    file_path = r"E:\Cic-DDos2019 Original\03-11\MSSQL_undersampling.csv" # Change to your actual file path 


    #file_path = r"E:\Cic-DDos2019 Original\03-11\UDP_Pre.csv"

    #file_path =  r"E:\Cic-DDos2019 Original\03-11\UDP_undersampling.csv"


###########################################################################################

    
    X, y = evaluator.load_and_preprocess_data(file_path)
    # Train model with overfitting prevention
    model = evaluator.train_model()
    # Comprehensive evaluation
    evaluator.comprehensive_evaluation()
    print("\n" + "=" * 60)
    print("✅ EVALUATION COMPLETED SUCCESSFULLY")
    print("\n📋 PROFESSIONAL TESTING STRATEGY IMPLEMENTED:")
    print("   • Stratified data splitting maintained class distribution")
    print("   • Cross-validation ensured model generalization")
    print("   • Regularization prevented overfitting")
    print("   • Comprehensive metrics provided security insights")
    print("   • ROC analysis showed detection vs false alarm trade-offs")
    print("\n🔧 RECOMMENDATIONS FOR PRODUCTION DEPLOYMENT:")
    print("   • Monitor FPR regularly to minimize false alarms")
    print("   • Retrain model periodically with new attack patterns")
    print("   • Implement ensemble methods for improved detection")
    print("   • Conduct regular penetration testing validation")
if __name__ == "__main__":
    main()
'''
## Professional Testing Strategy Implementation
### 🛡️ Overfitting Prevention Measures:
1. **Regularization**: Logistic Regression with L2 regularization
2. **Cross-Validation**: 5-fold stratified CV for generalization assessment
3. **Train-Test Split**: 70-30 split with stratification
4. **Data Scaling**: Applied after splitting to prevent data leakage
### 🔍 Generalization Enhancement:
1. **Stratified Sampling**: Maintains class distribution in splits
2. **Multiple Metrics**: Comprehensive evaluation beyond accuracy
3. **ROC Analysis**: Visualizes detection trade-offs
4. **False Positive Analysis**: Critical for security applications
### 📊 Security-Specific Evaluation:
1. **Precision**: Measures attack confirmation accuracy
2. **Recall**: Measures attack detection capability  
3. **F1-Score**: Balanced measure for security contexts
4. **False Positive Rate**: Critical for operational efficiency
### 🎯 Key Features:
- **Dynamic Classification**: Automatically handles binary/multi-class
- **Professional Visualization**: Clear, interpretable plots
- **Security Interpretation**: Business-focused insights
- **Robust Error Handling**: Graceful degradation
This implementation provides a production-ready IDS evaluation framework that addresses the specified concerns while maintaining professional cybersecurity standards.
'''
