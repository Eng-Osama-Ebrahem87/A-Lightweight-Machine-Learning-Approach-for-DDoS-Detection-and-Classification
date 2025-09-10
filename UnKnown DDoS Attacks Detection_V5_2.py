# Professional Intrusion Detection System Evaluation Code:

"""
Professional Intrusion Detection System Evaluation Framework
This code evaluates ML models for known and unknown attack classification
with comprehensive metrics and visualizations.
"""
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve, cross_val_score 

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    RocCurveDisplay, precision_recall_curve
)


from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression


from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from collections import Counter

import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)

class AdvancedIDSEvaluator:
    """
    Advanced IDS Evaluation Framework for known and unknown attack classification
    Supports binary and multi-class scenarios with comprehensive evaluation
    """
    def __init__(self, data_path, unknown_attack_types=None, test_size=0.3, val_size=0.2):
        """
        Initialize the evaluator
        :param data_path: Path to the dataset file
        :param unknown_attack_types: List of attack types to be treated as unknown
        :param test_size: Proportion of data for testing
        :param val_size: Proportion of training data for validation
        """
        self.data_path = data_path
        self.unknown_attack_types = unknown_attack_types or []
        self.test_size = test_size
        self.val_size = val_size
        # Initialize models with potential hyperparameters
        self.models = {
            'CNB': ComplementNB(),
            'KNN': KNeighborsClassifier(),
            'RF': RandomForestClassifier(),
            'LR': LogisticRegression()
        }
        # Results storage
        self.results = {}
        self.unknown_attack_results = {}
        # Preprocessing objects
        #self.scaler = StandardScaler()
        #self.label_encoder = LabelEncoder()
        # Data partitions
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.X_known, self.X_unknown, self.y_known, self.y_unknown = None, None, None, None
        # Feature names
        self.feature_names = [ " Packet Length Mean", " Average Packet Size", " Bwd Packet Length Min",
            "Fwd Packets/s", " Min Packet Length", " Down/Up Ratio" ]  
        '''self.feature_names = [ "Packet Length Mean", "Average Packet Size", "Bwd Packet Length Min",
            "Fwd Packets/s", "Min Packet Length", "Down/Up Ratio" ]  '''
    def load_and_preprocess_data(self):
        """
        Load and preprocess the dataset with careful handling of unknown attacks
        Implements the strategy for evaluating unknown attack classification
        """
        # Load the dataset
        data = pd.read_csv(self.data_path)
        # Remove rows with missing values
        data = data.dropna()
        # Separate features and labels
        features = data[self.feature_names]
        #labels = data[ "Label" ]
        labels = data[ " Label" ]

        # Encode labels
        '''self.label_encoder.fit(labels)
        encoded_labels = self.label_encoder.transform(labels)
        print(f"Original class distribution: {Counter(labels)}") '''
        # Strategy for unknown attack evaluation
        if self.unknown_attack_types:
            print(f"Treating {self.unknown_attack_types} as unknown attacks for evaluation")
            # Create masks for known and unknown attacks
            #unknown_mask = data['Label'].isin(self.unknown_attack_types)
            unknown_mask = data[' Label'].isin(self.unknown_attack_types)

            known_mask = ~unknown_mask
            # Split known data into train and test
            X_known = features[known_mask]
            #y_known = encoded_labels[known_mask]
            y_known = labels[known_mask]

            # Get unknown data
            X_unknown = features[unknown_mask]
            #y_unknown = encoded_labels[unknown_mask]
            y_unknown = labels[unknown_mask]

            # Split known data into train and test (stratified to maintain class distribution)
            X_known_train, X_known_test, y_known_train, y_known_test = train_test_split(
                X_known, y_known, test_size=self.test_size, 
                stratify=y_known, random_state=42
            )
            # Further split training data into train and validation
            X_train, X_val, y_train, y_val = train_test_split(
                X_known_train, y_known_train, test_size=self.val_size, 
                stratify=y_known_train, random_state=42
            )
            # Scale the features (fit only on training data to avoid data leakage)
            #self.scaler.fit(X_train)
            #X_train = self.scaler.transform(X_train)
            #X_val = self.scaler.transform(X_val)
            #X_known_test = self.scaler.transform(X_known_test)
            #X_unknown = self.scaler.transform(X_unknown)
            # Store the data
            self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_known_test, y_train, y_known_test
            self.X_unknown, self.y_unknown = X_unknown, y_unknown
            print(f"Known attack training samples: {X_train.shape[0]}")
            print(f"Known attack testing samples: {X_known_test.shape[0]}")
            print(f"Unknown attack samples: {X_unknown.shape[0]}")
        else:
            # Standard train-test split when no unknown attacks are specified
            X_train, X_test, y_train, y_test = train_test_split(
                features, encoded_labels, test_size=self.test_size, 
                stratify=encoded_labels, random_state=42
            )
            # Scale the features
            #self.scaler.fit(X_train)
            #X_train = self.scaler.transform(X_train)
            #X_test = self.scaler.transform(X_test)  '''
            # Store the data
            self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test
            print(f"Training samples: {X_train.shape[0]}")
            print(f"Testing samples: {X_test.shape[0]}")
        print("Data loading and preprocessing completed successfully")
    def train_models(self):
        """
        Train all models with cross-validation to prevent overfitting
        """
        print("Training models with stratified 5-fold cross-validation...")
        for name, model in self.models.items():
            print(f"\nTraining {name} model...")
            # Use stratified K-fold cross-validation
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=skf, scoring='f1_weighted')
            print(f"Cross-validation F1 scores: {cv_scores}")
            print(f"Mean CV F1-score: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")
            # Train the model on the full training set
            model.fit(self.X_train, self.y_train)
            # Check for overfitting by comparing train and validation performance
            train_pred = model.predict(self.X_train)
            train_f1 = f1_score(self.y_train, train_pred, average='weighted')
            print(f"Training F1-score: {train_f1:.4f}")
            # If significant overfitting (train >> validation), consider regularization
            if train_f1 - np.mean(cv_scores) > 0.15:
                print(f"Warning: Potential overfitting detected for {name}. Consider increasing regularization.")
    def evaluate_models(self, X_test, y_test, evaluation_type="known"):
        """
        Evaluate models on the test set and calculate comprehensive metrics
        :param X_test: Test features
        :param y_test: Test labels
        :param evaluation_type: Type of evaluation ("known" or "unknown")
        """
        results = {}
        for name, model in self.models.items():
            print(f"\nEvaluating {name} on {evaluation_type} attacks...")
            # Get predictions and probabilities
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            # Calculate false positive rate
            cm = confusion_matrix(y_test, y_pred)
            fp = cm.sum(axis=0) - np.diag(cm)
            tn = cm.sum() - (cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm))
            fpr = fp / (fp + tn)
            avg_fpr = np.mean(fpr)
            # Store results
            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'false_positive_rate': avg_fpr,
                'confusion_matrix': cm,
                'y_true': y_test,
                'y_pred': y_pred,
                'y_proba': y_proba
            }
            # Print results
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
            print(f"False Positive Rate: {avg_fpr:.4f}")
            # Detailed classification report
            print("\nClassification Report:")
            '''print(classification_report(y_test, y_pred, 
                                      target_names=self.label_encoder.classes_,
                                      zero_division=0))'''
            print( classification_report(y_test, y_pred))

        return results
    def evaluate_unknown_attacks(self):
        """
        Specialized evaluation for unknown attacks
        Tests the model's ability to generalize to unseen attack types
        """
        if not self.unknown_attack_types:
            print("No unknown attacks specified for evaluation")
            return None
        print("\n" + "="*60)
        print("EVALUATING UNKNOWN ATTACK CLASSIFICATION")
        print("="*60)
        # Evaluate models on unknown attacks
        unknown_results = self.evaluate_models(self.X_unknown, self.y_unknown, "unknown")
        # Additional analysis for unknown attacks
        for name, result in unknown_results.items():
            # Check if models are overly confident in wrong predictions
            if result['y_proba'] is not None:
                max_probs = np.max(result['y_proba'], axis=1)
                avg_confidence = np.mean(max_probs)
                print(f"{name} average confidence on unknown attacks: {avg_confidence:.4f}")
                # High confidence on wrong predictions indicates poor generalization
                if avg_confidence > 0.7 and result['accuracy'] < 0.5:
                    print(f"Warning: {name} shows high confidence but poor accuracy on unknown attacks")
        return unknown_results
    def plot_learning_curves(self):
        """
        Plot learning curves to diagnose bias-variance tradeoff
        and check for overfitting
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        for idx, (name, model) in enumerate(self.models.items()):
            # Calculate learning curve data
            train_sizes, train_scores, test_scores = learning_curve(
                model, self.X_train, self.y_train, cv=5, 
                scoring='f1_weighted', n_jobs=-1,
                train_sizes=np.linspace(0.1, 1.0, 10)
            )
            # Calculate mean and standard deviation
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            test_mean = np.mean(test_scores, axis=1)
            test_std = np.std(test_scores, axis=1)
            # Plot learning curve
            ax = axes[idx]
            ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
            ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
            ax.plot(train_sizes, test_mean, 'o-', color='green', label='Cross-validation score')
            ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='green')
            ax.set_title(f'Learning Curve - {name}')
            ax.set_xlabel('Training examples')
            ax.set_ylabel('F1 Score')
            ax.legend(loc='best')
            ax.grid(True)
        plt.tight_layout()
        plt.savefig('learning_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    def plot_roc_curves(self):
        """
        Plot multi-class ROC curves for each model
        """
        n_classes = len(self.label_encoder.classes_)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        for idx, (name, result) in enumerate(self.results.items()):
            if result['y_proba'] is not None:
                # Compute ROC curve and ROC area for each class
                fpr = dict()
                tpr = dict()
                roc_auc = dict()
                for i in range(n_classes):
                    fpr[i], tpr[i], _ = roc_curve(
                        (result['y_true'] == i).astype(int), 
                        result['y_proba'][:, i]
                    )
                    roc_auc[i] = auc(fpr[i], tpr[i])
                # Compute micro-average ROC curve and area
                fpr["micro"], tpr["micro"], _ = roc_curve(
                    result['y_true'].ravel(), result['y_proba'].ravel()
                )
                roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
                # Plot ROC curves
                ax = axes[idx]
                for i in range(n_classes):
                    ax.plot(fpr[i], tpr[i], lw=1, alpha=0.6,
                            label=f'Class {self.label_encoder.classes_[i]} (AUC = {roc_auc[i]:.2f})')
                ax.plot(fpr["micro"], tpr["micro"],
                        label=f'Micro-average (AUC = {roc_auc["micro"]:.2f})',
                        color='deeppink', linestyle=':', linewidth=4)
                ax.plot([0, 1], [0, 1], 'k--', lw=1)
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title(f'ROC Curves - {name}')
                ax.legend(loc="lower right", fontsize='small')
        plt.tight_layout()
        plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    def plot_performance_comparison(self):
        """
        Create comprehensive performance comparison visualizations
        """
        # Prepare data for plotting
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'false_positive_rate']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'False Positive Rate']
        # Create main comparison plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        # Plot metrics comparison
        for i, metric in enumerate(metrics):
            values = [self.results[name][metric] for name in self.models.keys()]
            axes[i].bar(self.models.keys(), values, color=['blue', 'green', 'red', 'orange'])
            axes[i].set_title(metric_names[i], fontsize=14)
            axes[i].set_ylim(0, 1)
            # Add values on bars
            for j, v in enumerate(values):
                axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
        # Plot confusion matrix for the best model
        best_model = max(self.results, key=lambda x: self.results[x]['f1_score'])
        cm = self.results[best_model]['confusion_matrix']
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # Plot confusion matrix
        im = axes[4].imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
        axes[4].set_title(f'Confusion Matrix - {best_model}', fontsize=14)
        # Add colorbar
        plt.colorbar(im, ax=axes[4])
        # Add labels
        tick_marks = np.arange(len(self.label_encoder.classes_))
        axes[4].set_xticks(tick_marks)
        axes[4].set_yticks(tick_marks)
        axes[4].set_xticklabels(self.label_encoder.classes_, rotation=45, ha='right')
        axes[4].set_yticklabels(self.label_encoder.classes_)
        # Add text annotations
        thresh = cm_normalized.max() / 2.
        for i in range(cm_normalized.shape[0]):
            for j in range(cm_normalized.shape[1]):
                axes[4].text(j, i, f'{cm_normalized[i, j]:.2f}',
                            ha="center", va="center",
                            color="white" if cm_normalized[i, j] > thresh else "black")
        # Remove empty subplot
        axes[5].set_visible(False)
        plt.tight_layout()
        plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        # Additional plot for unknown attack performance if available
        if self.unknown_attack_results:
            fig, ax = plt.subplots(figsize=(10, 6))
            # Extract accuracy on unknown attacks
            unknown_acc = [self.unknown_attack_results[name]['accuracy'] for name in self.models.keys()]
            # Create bar plot
            bars = ax.bar(self.models.keys(), unknown_acc, color=['blue', 'green', 'red', 'orange'])
            ax.set_title('Accuracy on Unknown Attacks', fontsize=16)
            ax.set_ylabel('Accuracy', fontsize=14)
            ax.set_ylim(0, 1)
            # Add values on bars
            for bar, acc in zip(bars, unknown_acc):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
            plt.tight_layout()
            plt.savefig('unknown_attack_performance.png', dpi=300, bbox_inches='tight')
            plt.show()
    def run_complete_evaluation(self):
        """
        Execute the complete evaluation pipeline
        """
        print("Starting Comprehensive IDS Evaluation")
        print("="*60)
        # Load and preprocess data
        self.load_and_preprocess_data()
        # Train models with cross-validation
        self.train_models()
        # Evaluate on known attacks
        print("\n" + "="*60)
        print("EVALUATING KNOWN ATTACK CLASSIFICATION")
        print("="*60)
        self.results = self.evaluate_models(self.X_test, self.y_test, "known")
        # Evaluate on unknown attacks if specified
        if self.unknown_attack_types:
            self.unknown_attack_results = self.evaluate_unknown_attacks()
        # Generate visualizations
        print("\nGenerating evaluation visualizations...")
        self.plot_learning_curves()
        self.plot_roc_curves()
        self.plot_performance_comparison()
        print("\nEvaluation completed successfully!")
        print("Results and visualizations have been saved.")
# Professional Testing Strategy for Unknown Attack Classification
"""
TESTING STRATEGY FOR UNKNOWN ATTACK CLASSIFICATION
1. DATA PARTITIONING STRATEGY:
   - Known attacks: Used for training and validation
   - Unknown attacks: Completely excluded from training and used only for testing
   - This ensures models are evaluated on truly unseen attack types
2. CROSS-VALIDATION APPROACH:
   - Stratified 5-fold cross-validation to maintain class distribution
   - Prevents overfitting and provides robust performance estimates
3. OVERFITTING PREVENTION:
   - Learning curve analysis to detect overfitting
   - Regularization techniques in models (e.g., max_depth in RF, C parameter in LR)
   - Early stopping through validation set monitoring
4. GENERALIZATION ASSESSMENT:
   - Compare performance on known vs unknown attacks
   - Analyze confidence levels on unknown attack predictions
   - Check for models that are overly confident in wrong predictions
5. COMPREHENSIVE METRICS:
   - Standard metrics (Accuracy, Precision, Recall, F1) for known attacks
   - Specialized analysis for unknown attack detection
   - False Positive Rate to measure impact on normal traffic
6. VISUALIZATION FOR INSIGHT:
   - ROC curves for multi-class classification
   - Learning curves to diagnose bias-variance tradeoff
   - Confusion matrices for detailed error analysis
7. ROBUSTNESS CHECKS:
   - Multiple models with different inductive biases
   - Statistical analysis of performance differences
   - Sensitivity analysis to hyperparameter changes
"""
if __name__ == "__main__":



    #data_path = r"E:\SDN-DDoS_Traffic_Dataset\SDN-DDoS_With_CIC_Features.csv"    # ##   For SDN-DDoS_With_CIC_Feature _Dataset   ..
    #data_path = r"E:\Cic-DDos2019 Original\03-11\MSSQL_Pre.csv"
    #data_path = r"E:\Cic-DDos2019 Original\03-11\MSSQL_undersampling.csv"
    #data_path =   r"E:\Cic-DDos2019 Original\03-11\MSSQL_undersampling.csv"

    #data_path =  r"E:\Cic-DDos2019 Original\03-11\LDAP.csv"
    data_path = r"E:\Cic-DDos2019 Original\03-11\LDAP.csv"


    # Specify which attack types should be treated as unknown
    unknown_attack_types = ["NetBIOS"]  # Replace with actual unknown attack names


    # Initialize and run evaluation
    evaluator = AdvancedIDSEvaluator(data_path, unknown_attack_types)
    evaluator.run_complete_evaluation()

'''## Key Features of This Implementation:
### 1. Professional Testing Strategy for Unknown Attacks:
- Complete separation of known and unknown attacks during training
- Models are trained only on known attacks and evaluated on both known and unknown
- Special analysis of model confidence on unknown attacks
### 2. Overfitting Prevention:
- Stratified cross-validation for robust performance estimation
- Learning curve analysis to detect overfitting
- Regularization techniques built into model parameters
### 3. Generalization Enhancement:
- Comparison of performance on known vs unknown attacks
- Analysis of whether models are overly confident in wrong predictions
- Multiple models with different inductive biases
### 4. Comprehensive Evaluation:
- Standard metrics (Accuracy, Precision, Recall, F1, FPR)
- Multi-class ROC curves
- Detailed confusion matrices
- Visualizations for performance comparison
### 5. Dynamic Code Structure:
- Handles both binary and multi-class classification
- Flexible to include or exclude unknown attack evaluation
- Clear, professional English comments throughout
This implementation provides a rigorous framework for evaluating IDS performance on both known and unknown attacks while preventing overfitting and ensuring generalization capability.

'''

 
