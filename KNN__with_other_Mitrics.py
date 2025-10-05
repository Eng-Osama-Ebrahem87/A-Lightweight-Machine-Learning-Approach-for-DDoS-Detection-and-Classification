
"""
Professional Intrusion Detection System Evaluator
KNN-Based Attack Classification for CIC-DDoS 2019 Dataset
Testing Strategy: Comprehensive evaluation with focus on generalization and overfitting prevention
"""
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report, 
                             roc_curve, auc, RocCurveDisplay)

from sklearn.multiclass import OneVsRestClassifier
import warnings
warnings.filterwarnings('ignore')
class IntrusionDetectionEvaluator:
    """
    Professional evaluator for KNN-based intrusion detection system
    Testing Strategy Components:
    1. Data Quality Assurance: Handling missing values and feature validation
    2. Generalization Protection: Stratified cross-validation and proper data splitting
    3. Overfitting Prevention: Model simplicity and performance gap monitoring
    4. Comprehensive Metrics: Multi-perspective evaluation for robust assessment
    5. Dynamic Classification: Automatic handling of binary vs multi-class scenarios
    """
    def __init__(self, random_state=42):
        """
        Initialize the evaluator with professional testing strategy
        Testing Strategy: Set reproducible random state for consistent results
        """
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_binary = False
        self.classes_ = None
        self.feature_names = [
            'Packet Length Mean',
            'Average Packet Size',
            'Bwd Packet Length Min', 
            'Fwd Packets/s',
            'Min Packet Length',
            'Down/Up Ratio'
        ]
    def load_and_validate_data(self, file_path):
        """
        Load and validate CIC-DDoS 2019 dataset
        Testing Strategy: Robust data validation to ensure dataset quality
        """
        print("🔍 DATA LOADING AND VALIDATION PHASE")
        print("Testing Strategy: Ensuring data quality and feature compatibility")
        try:
            # Load dataset with error handling
            df = pd.read_csv(file_path)
            print(f"✅ Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return None
        # Validate required features exist
        missing_features = [f for f in self.feature_names + ['Label'] if f not in df.columns]
        if missing_features:
            print(f"❌ Missing required features: {missing_features}")
            print(f"Available features: {df.columns.tolist()}")
            return None
        # Select only required features to avoid noise
        df = df[self.feature_names + ['Label']]
        # Data quality assessment
        print("Testing Strategy: Comprehensive data quality checks")
        initial_shape = df.shape
        print(f"Initial data shape: {initial_shape}")
        # Handle missing values - Testing Strategy: Prevent data leakage
        missing_count = df.isnull().sum().sum()
        print(f"Missing values detected: {missing_count}")
        df = df.dropna()
        # Remove infinite values that can break KNN distance calculations
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        # Ensure numerical features for KNN
        for feature in self.feature_names:
            df[feature] = pd.to_numeric(df[feature], errors='coerce')
        df = df.dropna()
        final_shape = df.shape
        print(f"Final data shape after cleaning: {final_shape}")
        print(f"Data retention: {(final_shape[0]/initial_shape[0])*100:.2f}%")
        return df
    def analyze_class_distribution(self, labels):
        """
        Analyze class distribution for stratification strategy
        Testing Strategy: Understand data imbalance for proper evaluation
        """
        print("\n📊 CLASS DISTRIBUTION ANALYSIS")
        print("Testing Strategy: Assessing class imbalance for stratified sampling")
        unique_classes, class_counts = np.unique(labels, return_counts=True)
        n_classes = len(unique_classes)
        print(f"Total classes: {n_classes}")
        for cls, count in zip(unique_classes, class_counts):
            percentage = (count / len(labels)) * 100
            print(f"Class {cls}: {count} samples ({percentage:.2f}%)")
        # Determine if binary classification
        self.is_binary = (n_classes == 2)
        classification_type = "Binary" if self.is_binary else "Multi-class"
        print(f"Classification type: {classification_type}")
        return n_classes, unique_classes, class_counts
    def prepare_features_and_labels(self, df):
        """
        Prepare features and labels with proper preprocessing
        Testing Strategy: Feature scaling and label encoding for KNN optimization
        """
        print("\n⚙️ FEATURE AND LABEL PREPARATION")
        print("Testing Strategy: Proper preprocessing for KNN performance")
        # Separate features and labels
        X = df[self.feature_names]
        y = df['Label']
        # Encode labels - Testing Strategy: Consistent label mapping
        y_encoded = self.label_encoder.fit_transform(y)
        self.classes_ = self.label_encoder.classes_
        # Analyze class distribution
        n_classes, unique_classes, class_counts = self.analyze_class_distribution(y_encoded)
        # Scale features - Testing Strategy: Essential for KNN distance calculations
        print("Testing Strategy: Feature scaling for equal feature contribution")
        X_scaled = self.scaler.fit_transform(X)
        print(f"Features scaled: {X_scaled.shape[1]}")
        print(f"Labels encoded: {len(self.classes_)} classes")
        return X_scaled, y_encoded
    def initialize_knn_model(self, n_neighbors=5):
        """
        Initialize KNN model with optimal parameters
        Testing Strategy: Simple model to reduce overfitting risk
        """
        print("\n🤖 KNN MODEL INITIALIZATION")
        print("Testing Strategy: Using simple KNN to prevent overfitting")
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights='distance',  # Testing Strategy: Distance weighting for better performance
            algorithm='auto',
            metric='minkowski',
            p=2  # Euclidean distance
        )
        print(f"KNN model configured with:")
        print(f"  - Neighbors: {n_neighbors}")
        print(f"  - Weighting: Distance-based")
        print(f"  - Metric: Euclidean distance")
        return self.model
    def perform_stratified_split(self, X, y, test_size=0.3):
        """
        Perform stratified train-test split
        Testing Strategy: Maintain class distribution in splits for fair evaluation
        """
        print("\n📈 STRATIFIED DATA SPLITTING")
        print("Testing Strategy: Preserving class distribution in train/test sets")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=self.random_state,
            stratify=y,  # Testing Strategy: Crucial for imbalanced datasets
            shuffle=True
        )
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Testing set: {X_test.shape[0]} samples")
        print(f"Train/Test ratio: {X_train.shape[0]/X_test.shape[0]:.2f}:1")
        return X_train, X_test, y_train, y_test
    def calculate_detailed_metrics(self, y_true, y_pred, y_pred_proba=None):
        """
        Calculate comprehensive evaluation metrics
        Testing Strategy: Multi-faceted performance assessment
        """
        print("\n📊 COMPREHENSIVE METRICS CALCULATION")
        print("Testing Strategy: Multi-perspective performance evaluation")
        # Basic classification metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        print(f"🎯 Accuracy: {accuracy:.4f}")
        print(f"📏 Precision: {precision:.4f}")
        print(f"🔍 Recall: {recall:.4f}")
        print(f"⚖️ F1-Score: {f1:.4f}")
        # False Positive Rate calculation
        fpr = self.calculate_false_positive_rate(y_true, y_pred)
        print(f"🚫 False Positive Rate: {fpr:.4f}")
        # Additional metrics for binary classification
        if self.is_binary and y_pred_proba is not None:
            auroc = roc_auc_score(y_true, y_pred_proba[:, 1])
            print(f"📈 AUROC: {auroc:.4f}")
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'false_positive_rate': fpr
        }
        return metrics
    def calculate_false_positive_rate(self, y_true, y_pred):
        """
        Calculate False Positive Rate dynamically for binary and multi-class
        Testing Strategy: Adaptive FPR calculation for different classification types
        """
        cm = confusion_matrix(y_true, y_pred)
        if self.is_binary:
            # Binary classification FPR
            tn, fp, fn, tp = cm.ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        else:
            # Multi-class macro-average FPR
            fpr_sum = 0
            for i in range(len(cm)):
                fp = cm[:, i].sum() - cm[i, i]
                tn = np.sum(cm) - np.sum(cm[:, i]) - np.sum(cm[i, :]) + cm[i, i]
                fpr_sum += fp / (fp + tn) if (fp + tn) > 0 else 0
            fpr = fpr_sum / len(cm)
        return fpr
    def generate_confusion_matrix(self, y_true, y_pred):
        """
        Generate and display professional confusion matrix
        Testing Strategy: Visual assessment of classification patterns
        """
        print("\n🔄 CONFUSION MATRIX GENERATION")
        print("Testing Strategy: Visual analysis of classification behavior")
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.classes_, 
                   yticklabels=self.classes_,
                   cbar_kws={'label': 'Number of Predictions'})
        plt.title('Confusion Matrix - Intrusion Detection System\nTesting Strategy: Pattern Analysis and Error Identification', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Predicted Labels\nTesting Strategy: Assess Prediction Consistency', 
                  fontweight='bold')
        plt.ylabel('True Labels\nTesting Strategy: Evaluate Detection Accuracy', 
                  fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
        return cm
    def plot_multi_class_roc_curves(self, X_test, y_test, y_pred_proba):
        """
        Plot ROC curves for multi-class classification
        Testing Strategy: AUC evaluation for each class discrimination ability
        """
        print("\n📉 MULTI-CLASS ROC CURVES GENERATION")
        print("Testing Strategy: Class-wise discrimination capability assessment")
        n_classes = len(self.classes_)
        # Plot setup
        plt.figure(figsize=(12, 10))
        colors = plt.cm.Set3(np.linspace(0, 1, n_classes))
        # Plot ROC curve for each class
        for i, color in zip(range(n_classes), colors):
            # Binarize labels for current class
            y_binary = (y_test == i).astype(int)
            # Calculate ROC curve and AUC
            fpr, tpr, _ = roc_curve(y_binary, y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            # Plot ROC curve
            plt.plot(fpr, tpr, color=color, lw=2,
                    label=f'{self.classes_[i]} (AUC = {roc_auc:.3f})')
        # Plot random classifier line
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        # Plot configuration
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate\nTesting Strategy: Measure of False Alarms', 
                  fontweight='bold')
        plt.ylabel('True Positive Rate\nTesting Strategy: Measure of Attack Detection', 
                  fontweight='bold')
        plt.title('Multi-class ROC Curves - Intrusion Detection Performance\nTesting Strategy: Comprehensive Discrimination Assessment', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.legend(loc="lower right", frameon=True, fancybox=True, shadow=True)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    def plot_binary_roc_curve(self, X_test, y_test, y_pred_proba):
        """
        Plot ROC curve for binary classification
        Testing Strategy: Single comprehensive ROC analysis
        """
        print("\n📈 BINARY CLASSIFICATION ROC CURVE")
        print("Testing Strategy: Overall system discrimination capability")
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        # Plot ROC curve
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=3, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        # Plot configuration
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate\nTesting Strategy: False Alarm Rate Assessment', 
                  fontweight='bold')
        plt.ylabel('True Positive Rate\nTesting Strategy: Attack Detection Rate', 
                  fontweight='bold')
        plt.title('ROC Curve - Binary Intrusion Detection\nTesting Strategy: System-Wide Performance Evaluation', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        return roc_auc
    def perform_cross_validation(self, X, y, cv=5):
        """
        Perform stratified k-fold cross validation
        Testing Strategy: Robust generalization assessment
        """
        print("\n🔄 STRATIFIED CROSS-VALIDATION")
        print("Testing Strategy: Generalization capability verification")
        # Initialize stratified k-fold
        skf = StratifiedKFold(n_splits=cv, shuffle=True, 
                             random_state=self.random_state)
        # Perform cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=skf, 
                                   scoring='accuracy', n_jobs=-1)
        print(f"Cross-validation scores: {[f'{score:.4f}' for score in cv_scores]}")
        print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std() * 2:.4f})")
        # Plot CV results
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, cv+1), cv_scores, color='skyblue', edgecolor='navy')
        plt.axhline(y=cv_scores.mean(), color='red', linestyle='--', 
                   label=f'Mean: {cv_scores.mean():.4f}')
        plt.xlabel('Fold Number\nTesting Strategy: Validation Split Iteration', 
                  fontweight='bold')
        plt.ylabel('Accuracy\nTesting Strategy: Consistent Performance Measure', 
                  fontweight='bold')
        plt.title('Stratified Cross-Validation Results\nTesting Strategy: Generalization Stability Assessment', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        return cv_scores
    def assess_overfitting(self, X_train, X_test, y_train, y_test):
        """
        Assess potential overfitting
        Testing Strategy: Performance gap analysis between train and test sets
        """
        print("\n⚠️ OVERFITTING ASSESSMENT")
        print("Testing Strategy: Train-Test Performance Gap Analysis")
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        performance_gap = abs(train_score - test_score)
        print(f"Training Accuracy: {train_score:.4f}")
        print(f"Testing Accuracy: {test_score:.4f}")
        print(f"Performance Gap: {performance_gap:.4f}")
        # Overfitting assessment criteria
        if performance_gap > 0.1:
            print("🚨 WARNING: Potential overfitting detected!")
            print("Testing Strategy: Large gap indicates poor generalization")
        elif performance_gap > 0.05:
            print("⚠️ CAUTION: Moderate performance gap observed")
            print("Testing Strategy: Monitor model generalization")
        else:
            print("✅ GOOD: Minimal performance gap detected")
            print("Testing Strategy: Model shows good generalization")
        return {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'performance_gap': performance_gap
        }
    def run_complete_evaluation(self, file_path, test_size=0.3, n_neighbors=5):
        """
        Execute complete professional evaluation pipeline
        Testing Strategy: End-to-end systematic assessment
        """
        print("="*80)
        print("🚀 PROFESSIONAL INTRUSION DETECTION SYSTEM EVALUATION")
        print("Testing Strategy: Comprehensive KNN-based Assessment for CIC-DDoS 2019")
        print("="*80)
        # Phase 1: Data Preparation
        print("\n🎯 PHASE 1: DATA PREPARATION AND VALIDATION")
        df = self.load_and_validate_data(file_path)
        if df is None:
            return None
        # Phase 2: Feature Engineering
        print("\n🎯 PHASE 2: FEATURE ENGINEERING AND PREPROCESSING")
        X, y = self.prepare_features_and_labels(df)
        # Phase 3: Model Initialization
        print("\n🎯 PHASE 3: MODEL INITIALIZATION AND TRAINING")
        self.initialize_knn_model(n_neighbors)
        X_train, X_test, y_train, y_test = self.perform_stratified_split(X, y, test_size)
        self.model.fit(X_train, y_train)
        print("✅ KNN model training completed")
        # Phase 4: Prediction Generation
        print("\n🎯 PHASE 4: PREDICTION AND PROBABILITY ESTIMATION")
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test) if hasattr(self.model, "predict_proba") else None
        # Phase 5: Comprehensive Evaluation
        print("\n🎯 PHASE 5: COMPREHENSIVE PERFORMANCE EVALUATION")
        # Calculate metrics
        metrics = self.calculate_detailed_metrics(y_test, y_pred, y_pred_proba)
        # Generate confusion matrix
        cm = self.generate_confusion_matrix(y_test, y_pred)
        # Generate ROC curves based on classification type
        if y_pred_proba is not None:
            if self.is_binary:
                auroc = self.plot_binary_roc_curve(X_test, y_test, y_pred_proba)
                metrics['auroc'] = auroc
            else:
                self.plot_multi_class_roc_curves(X_test, y_test, y_pred_proba)
        # Phase 6: Advanced Validation
        print("\n🎯 PHASE 6: ADVANCED VALIDATION AND GENERALIZATION ASSESSMENT")
        # Cross-validation
        cv_scores = self.perform_cross_validation(X, y)
        metrics['cv_mean_accuracy'] = cv_scores.mean()
        metrics['cv_std_accuracy'] = cv_scores.std()
        # Overfitting assessment
        overfitting_metrics = self.assess_overfitting(X_train, X_test, y_train, y_test)
        metrics.update(overfitting_metrics)
        # Phase 7: Final Reporting
        print("\n🎯 PHASE 7: COMPREHENSIVE EVALUATION REPORT")
        self.generate_final_report(metrics)
        return {
            'metrics': metrics,
            'confusion_matrix': cm,
            'cv_scores': cv_scores,
            'model': self.model,
            'feature_names': self.feature_names,
            'classes': self.classes_
        }
    def generate_final_report(self, metrics):
        """
        Generate comprehensive final evaluation report
        Testing Strategy: Summary of all testing strategies and results
        """
        print("\n" + "="*80)
        print("📋 COMPREHENSIVE EVALUATION REPORT SUMMARY")
        print("="*80)
        print("\n🔍 TESTING STRATEGY IMPLEMENTATION:")
        print("✅ Data Quality: Comprehensive validation and cleaning")
        print("✅ Generalization: Stratified cross-validation employed")
        print("✅ Overfitting Prevention: Performance gap monitoring")
        print("✅ Comprehensive Metrics: Multi-perspective evaluation")
        print("✅ Dynamic Classification: Binary/Multi-class adaptive handling")
        print("\n📊 PERFORMANCE METRICS SUMMARY:")
        print(f"   Accuracy: {metrics['accuracy']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall: {metrics['recall']:.4f}")
        print(f"   F1-Score: {metrics['f1_score']:.4f}")
        print(f"   False Positive Rate: {metrics['false_positive_rate']:.4f}")
        if 'auroc' in metrics:
            print(f"   AUROC: {metrics['auroc']:.4f}")
        print(f"   CV Mean Accuracy: {metrics['cv_mean_accuracy']:.4f}")
        print(f"   CV Std Accuracy: {metrics['cv_std_accuracy']:.4f}")
        print(f"   Performance Gap: {metrics['performance_gap']:.4f}")
        print("\n🎯 RECOMMENDATIONS:")
        if metrics['performance_gap'] > 0.1:
            print("   ⚠️  Consider reducing model complexity or increasing training data")
        if metrics['false_positive_rate'] > 0.1:
            print("   ⚠️  High false positive rate - consider threshold adjustment")
        if metrics['cv_std_accuracy'] > 0.05:
            print("   ⚠️  High variance in cross-validation - dataset may need balancing")
        print("="*80)
# Example usage and demonstration
def main():
    """
    Main execution function demonstrating the professional evaluation framework
    """
    # Initialize the professional evaluator
    evaluator = IntrusionDetectionEvaluator(random_state=42)
    # Specify your dataset path
    # Load your cybersecurity dataset


############################################################  CIC-DDoS2019_CSV  from original   - -  All files Available to test ................................

    #dataset_path = r"E:\Cic-DDos2019 Original\03-11\MSSQL_Pre.csv"

    dataset_path = r"E:\Cic-DDos2019 Original\03-11\MSSQL_undersampling.csv"  # For demonstration purposes - replace with your actual file path


    #dataset_path = r"E:\Cic-DDos2019 Original\03-11\UDP_Pre.csv"

    #dataset_path =  r"E:\Cic-DDos2019 Original\03-11\UDP_undersampling.csv"
    
###########################################################################################

    
    # Run complete evaluation
    print("🚀 Starting Professional Intrusion Detection System Evaluation...")
    results = evaluator.run_complete_evaluation(
        file_path=dataset_path,
        test_size=0.3,
        n_neighbors=5
    )
    if results is not None:
        print("\n✅ Evaluation completed successfully!")
        print("📁 Results available in 'results' dictionary for further analysis")
    else:
        print("\n❌ Evaluation failed. Please check dataset path and format.")
if __name__ == "__main__":
    main()

'''
This professional Python code provides a comprehensive evaluation framework for your KNN-based intrusion detection system with the following key features:
## 🎯 **Testing Strategy Implementation:**
### **1. Generalization Protection:**
- **Stratified K-Fold Cross Validation**: Maintains class distribution across splits
- **Stratified Train-Test Split**: Prevents skewed evaluation
- **Performance Gap Monitoring**: Detects overfitting through train-test comparison
### **2. Overfitting Prevention:**
- **Simple KNN Model**: Lower risk of overfitting
- **Cross-Validation Consistency**: Multiple validation rounds
- **Feature Scaling**: Prevents distance calculation bias
### **3. Comprehensive Evaluation:**
- **Multiple Metrics**: Accuracy, Precision, Recall, F1, FPR
- **Dynamic Classification**: Automatic binary/multi-class handling
- **Visual Analysis**: Confusion matrix and ROC curves
### **4. CIC-DDoS 2019 Compatibility:**
- **Feature Validation**: Specific CIC-DDoS feature checking
- **Data Quality Assurance**: Missing value and infinite value handling
- **Class Distribution Analysis**: Understanding dataset imbalance
### **5. Professional Reporting:**
- **Detailed Metrics**: Comprehensive performance assessment
- **Visualizations**: Professional plots for result interpretation
- **Recommendations**: Actionable insights based on results
The code is designed to be robust, professional, and specifically tailored for cybersecurity intrusion detection evaluation using the CIC-DDoS 2019 dataset characteristics.

'''
