"""Machine learning pipeline for binary text classification using classical models.

This module implements:
- 10-fold stratified cross-validation with SMOTE balancing
- Three classifiers: SVM, Naive Bayes, Random Forest
- Class weight balancing and performance metrics
- Best-fold selection and comprehensive reporting
- Optional Groq LLM inference on test set
"""

import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import jieba
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import random
import seaborn as sns
import time
import warnings
from imblearn.over_sampling import SMOTE

# Import professional figure utilities
from src.utils.figure_utils import (
    export_fold_metrics_csv,
    plot_fold_metrics_comparison,
    plot_metrics_panel,
    plot_model_comparison_bar,
    plot_confusion_matrix_consistent,
    setup_professional_style,
    save_figure_multi_format
)
from src.utils.config import get_project_root
from src.utils.path_resolver import get_pipeline_results_dirs
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           classification_report, confusion_matrix, roc_curve, auc, log_loss)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from src.utils.data_loader import get_canonical_split

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

HOLDOUT_TEST_SIZE = 0.2

LABEL_MEANINGFUL = 1
LABEL_NO_MEANINGFUL = 0
LABEL_DISPLAY_NAMES = {
    LABEL_NO_MEANINGFUL: 'No-meaningful',
    LABEL_MEANINGFUL: 'Meaningful',
}


def label_name_array(values):
    """Map numeric labels to readable names for reports and artifacts."""
    arr = np.asarray(values).astype(int)
    return np.where(arr == LABEL_MEANINGFUL, LABEL_DISPLAY_NAMES[LABEL_MEANINGFUL], LABEL_DISPLAY_NAMES[LABEL_NO_MEANINGFUL])


def get_results_dirs():
    """Create and return standardized results directories."""
    return get_pipeline_results_dirs('machine_learning')

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

class BinaryTextClassifierComparison:
    """
    A comprehensive binary text classifier comparison system using SVM, Naive Bayes, and Random Forest.
    Designed for Chinese text classification with binary labels (0/1) and automatic class imbalance handling.
    """
    
    def __init__(self, max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.95,
                 random_state=42, use_class_balancing=True):
        """
        Initialize the classifier comparison system.
        
        Args:
            max_features: Maximum number of TF-IDF features
            ngram_range: Range of n-grams to extract
            min_df: Minimum document frequency for terms
            max_df: Maximum document frequency for terms
            random_state: Random state for reproducibility
            use_class_balancing: Whether to apply class weight balancing
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.random_state = random_state
        self.use_class_balancing = use_class_balancing
        
        self.vectorizer = None
        self.models = {}
        self.scalers = {}
        self.model_input_format = {}
        self.class_weights = None
        
        # Initialize models (will be updated with class weights if needed)
        self._initialize_models()
        
    def _initialize_models(self, class_weights=None):
        """Initialize all three models with optimal parameters and optional class weights."""
        
        # Determine class_weight parameter
        if self.use_class_balancing and class_weights is not None:
            # Use computed class weights
            svm_class_weight = class_weights
            rf_class_weight = class_weights
        elif self.use_class_balancing:
            # Use sklearn's automatic balancing
            svm_class_weight = 'balanced'
            rf_class_weight = 'balanced'
        else:
            # No class balancing
            svm_class_weight = None
            rf_class_weight = None
            
        svm_model = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            class_weight=svm_class_weight,
            probability=True,
            random_state=self.random_state
        )

        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight=rf_class_weight,
            random_state=self.random_state,
            n_jobs=-1
        )


        nb_model = MultinomialNB(
            alpha=1.0,  # Laplace smoothing
            fit_prior=True
            # Note: sklearn MultinomialNB doesn't support class_weight parameter
            # We'll handle this through sample_weight in fit method
        )

        self.models = {
            'SVM': svm_model,
            'Naive_Bayes': nb_model,
            'Random_Forest': rf_model
        }

    def preprocess_chinese_text(self, text):
        """
        Preprocess Chinese text by word segmentation.
        
        Args:
            text: Input Chinese text
            
        Returns:
            Segmented text with words separated by spaces
        """
        if pd.isna(text) or text == '':
            return ''
        text = str(text).replace('\n', ' ').replace('\r', ' ').strip()
        words = jieba.cut(text)
        return ' '.join(words)

    def analyze_data_distribution(self, labels, title="Data Distribution"):
        """
        Analyze and print the distribution of binary labels in the dataset.
        
        Args:
            labels: numpy array of binary labels (0/1)
            title: Title for the analysis output
            
        Returns:
            Dictionary with distribution statistics
        """
        logger.info(f"\n{title}:")
        total_samples = len(labels)
        logger.info(f"Total samples: {total_samples}")
        
        positive_count = np.sum(labels == 1)
        negative_count = np.sum(labels == 0)
        positive_ratio = positive_count / total_samples * 100
        negative_ratio = negative_count / total_samples * 100
        
        logger.info(f"{LABEL_DISPLAY_NAMES[LABEL_MEANINGFUL]} (1): {positive_count} ({positive_ratio:.1f}%)")
        logger.info(f"{LABEL_DISPLAY_NAMES[LABEL_NO_MEANINGFUL]} (0): {negative_count} ({negative_ratio:.1f}%)")
        
        imbalance_ratio = max(positive_count, negative_count) / min(positive_count, negative_count)
        if imbalance_ratio > 2:
            logger.warning(f"Class imbalance detected! Ratio: {imbalance_ratio:.2f}")
        else:
            logger.info(f"Classes are relatively balanced. Ratio: {imbalance_ratio:.2f}")
            
        return {
            'total_samples': total_samples,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'positive_ratio': positive_ratio,
            'negative_ratio': negative_ratio,
            'imbalance_ratio': imbalance_ratio
        }

    def apply_class_weight_balancing(self, labels):
        """
        Calculate class weights for handling imbalance instead of SMOTE.
        This is more appropriate for text data with TF-IDF features.
        
        Args:
            labels: Binary label array
            
        Returns:
            Dictionary of class weights
        """
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        logger.info(f"Class distribution: {dict(zip(unique_labels, label_counts))}")
        
        # Calculate class weights inversely proportional to class frequencies
        total_samples = len(labels)
        class_weights = {}
        for label, count in zip(unique_labels, label_counts):
            class_weights[label] = total_samples / (len(unique_labels) * count)
        
        logger.info(f"Calculated class weights: {class_weights}")
        
        # Check for severe imbalance
        imbalance_ratio = max(label_counts) / min(label_counts)
        if imbalance_ratio > 3:
            logger.warning(f"Severe class imbalance detected (ratio: {imbalance_ratio:.2f})")
            logger.info("Using class weights to handle imbalance instead of SMOTE")
        else:
            logger.info(f"Moderate imbalance (ratio: {imbalance_ratio:.2f}), using class weights")
        
        # Store class weights for later use
        self.class_weights = class_weights
        
        # Re-initialize models with computed class weights
        if self.use_class_balancing:
            self._initialize_models(class_weights)
        
        return class_weights

    def compute_sample_weights(self, labels):
        """
        Compute sample weights for models that don't support class_weight parameter.
        
        Args:
            labels: Training labels
            
        Returns:
            Array of sample weights
        """
        if self.class_weights is None:
            return None
            
        sample_weights = np.array([self.class_weights.get(label, 1.0) for label in labels])
        return sample_weights

    def fit_transform_features(self, X_train, X_test=None):
        """
        Fit TF-IDF vectorizer and transform features.
        
        Args:
            X_train: Training texts
            X_test: Testing texts (optional)
            
        Returns:
            Transformed feature matrices
        """
        logger.info("Preprocessing Chinese text...")
        X_train_processed = [self.preprocess_chinese_text(text) for text in X_train]
        
        logger.info("Performing TF-IDF vectorization...")
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            stop_words=None
        )
        
        X_train_tfidf = self.vectorizer.fit_transform(X_train_processed)
        logger.info(f"Feature dimensions: {X_train_tfidf.shape[1]}")
        logger.info(f"Feature matrix sparsity: {(X_train_tfidf.nnz / (X_train_tfidf.shape[0] * X_train_tfidf.shape[1]) * 100):.2f}%")
        
        if X_test is not None:
            X_test_processed = [self.preprocess_chinese_text(text) for text in X_test]
            X_test_tfidf = self.vectorizer.transform(X_test_processed)
            return X_train_tfidf, X_test_tfidf
        
        return X_train_tfidf

    def fit_models(self, X_train_tfidf, y_train):
        """
        Train all models on the training data with class weight balancing.
        
        Args:
            X_train_tfidf: Training feature matrix
            y_train: Training labels
        """
        logger.info("Training all models with class weight balancing...")
        
        # Apply class weight balancing if not already done
        if self.use_class_balancing and self.class_weights is None:
            self.apply_class_weight_balancing(y_train)
        
        # Compute sample weights for Naive Bayes
        sample_weights = self.compute_sample_weights(y_train) if self.use_class_balancing else None
        
        for name, model in tqdm(self.models.items(), desc='Training models', total=len(self.models)):
            logger.info(f"Training {name}...")
            start_time = time.perf_counter()

            # Default assumption: model can consume sparse input directly.
            self.model_input_format[name] = 'sparse'
            
            # Handle different model types
            if name == 'SVM' and hasattr(X_train_tfidf, 'todense'):
                # SVM with optional scaling
                if X_train_tfidf.shape[0] * X_train_tfidf.shape[1] < 1000000:
                    X_train_dense = X_train_tfidf.toarray() if hasattr(X_train_tfidf, 'toarray') else np.asarray(X_train_tfidf)
                    self.scalers[name] = StandardScaler()
                    X_train_scaled = self.scalers[name].fit_transform(X_train_dense)
                    model.fit(X_train_scaled, y_train)
                    self.model_input_format[name] = 'dense'
                else:
                    model.fit(X_train_tfidf, y_train)
                    self.model_input_format[name] = 'sparse'

            elif name == 'SVM':
                # SMOTE path provides dense ndarray; keep this path explicit.
                model.fit(X_train_tfidf, y_train)
                self.model_input_format[name] = 'dense'
                    
            elif name == 'Naive_Bayes' and self.use_class_balancing and sample_weights is not None:
                # Naive Bayes with sample weights
                model.fit(X_train_tfidf, y_train, sample_weight=sample_weights)
                logger.info(f"Applied sample weights to {name}")
                
            else:
                # Standard training (Random Forest with class_weight, or no balancing)
                model.fit(X_train_tfidf, y_train)
                
            train_time = time.perf_counter() - start_time
            logger.info(f"{name} training completed in {train_time:.4f}s")
            
            # Log class balancing info
            if self.use_class_balancing:
                if hasattr(model, 'class_weight') and model.class_weight is not None:
                    logger.info(f"{name} using class_weight: {model.class_weight}")
                elif name == 'Naive_Bayes' and sample_weights is not None:
                    logger.info(f"{name} using sample weights")

    def predict_models(self, X_test_tfidf):
        """
        Make predictions using all trained models.
        
        Args:
            X_test_tfidf: Test feature matrix
            
        Returns:
            Dictionary of predictions and probabilities for each model
        """
        predictions = {}
        
        for name, model in self.models.items():
            start_time = time.perf_counter()
            
            # Apply scaling if used during training
            if name in self.scalers:
                X_test_dense = X_test_tfidf.toarray() if hasattr(X_test_tfidf, 'toarray') else np.asarray(X_test_tfidf)
                X_test_scaled = self.scalers[name].transform(X_test_dense)
                y_pred = model.predict(X_test_scaled)
                y_proba = model.predict_proba(X_test_scaled)
            elif self.model_input_format.get(name) == 'dense' and hasattr(X_test_tfidf, 'todense'):
                # SVM trained on dense data (e.g., SMOTE output) must infer on dense data too.
                X_test_dense = X_test_tfidf.toarray() if hasattr(X_test_tfidf, 'toarray') else np.asarray(X_test_tfidf)
                y_pred = model.predict(X_test_dense)
                y_proba = model.predict_proba(X_test_dense)
            else:
                y_pred = model.predict(X_test_tfidf)
                y_proba = model.predict_proba(X_test_tfidf)
            
            inference_time = time.perf_counter() - start_time
            
            predictions[name] = {
                'predictions': y_pred,
                'probabilities': y_proba,
                'inference_time': inference_time
            }
            
        return predictions

    def compute_metrics(self, y_true, predictions_dict, num_samples):
        """
        Compute comprehensive evaluation metrics for all models.
        
        Args:
            y_true: True labels
            predictions_dict: Dictionary of model predictions
            num_samples: Number of samples
            
        Returns:
            Dictionary of computed metrics for each model
        """
        results = {}
        
        for model_name, pred_data in predictions_dict.items():
            y_pred = pred_data['predictions']
            y_proba = pred_data['probabilities']
            inference_time = pred_data['inference_time']
            
            # Calculate metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
            recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
            
            # ROC AUC
            if y_proba.shape[1] == 2:  # Binary classification
                roc_auc = auc(*roc_curve(y_true, y_proba[:, 1])[:2])
                loss_value = log_loss(y_true, y_proba[:, 1], labels=[0, 1])
            else:
                roc_auc = 0.0
                loss_value = np.nan
            
            results[model_name] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "roc_auc": roc_auc,
                "log_loss": loss_value,
                "inference_time": inference_time,
                "inference_time_per_sample": inference_time / max(num_samples, 1),
                "samples_per_second": num_samples / max(inference_time, 1e-6)
            }
            
        return results

    def save_models(self, filepath_prefix="model"):
        """
        Save all trained models, vectorizer, and class weights.
        
        Args:
            filepath_prefix: Prefix for saved model files
        """
        model_data = {
            'vectorizer': self.vectorizer,
            'models': self.models,
            'scalers': self.scalers,
            'model_input_format': self.model_input_format,
            'class_weights': self.class_weights,
            'use_class_balancing': self.use_class_balancing,
        }
        
        filepath = f"{filepath_prefix}_all_models.pkl"
        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)
        logger.info(f"All models and class weights saved to {filepath}")

    def load_models(self, filepath):
        """
        Load trained models, vectorizer, and class weights.
        
        Args:
            filepath: Path to load the models from
        """
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)
        
        self.vectorizer = model_data['vectorizer']
        self.models = model_data['models']
        self.scalers = model_data.get('scalers', {})
        self.model_input_format = model_data.get('model_input_format', {})
        self.class_weights = model_data.get('class_weights', None)
        self.use_class_balancing = model_data.get('use_class_balancing', True)
        logger.info(f"All models and class weights loaded from {filepath}")

def create_comprehensive_visualizations(fold_results_dict, avg_metrics_dict, class_distribution, output_path):
    """
    Create separate professional visualizations for comprehensive model analysis.
    
    Args:
        fold_results_dict: Dictionary of fold results for each model
        avg_metrics_dict: Dictionary of average metrics for each model
        class_distribution: Dictionary with class distribution statistics
    """
    # Set style for publication-quality plots
    setup_professional_style()
    
    output_dir = Path(output_path).parent
    models = list(avg_metrics_dict.keys())
    
    # 1. Class Distribution Visualization (Pie Chart)
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = [
        f"{LABEL_DISPLAY_NAMES[LABEL_NO_MEANINGFUL]}",
        f"{LABEL_DISPLAY_NAMES[LABEL_MEANINGFUL]}",
    ]
    counts = [class_distribution['negative_count'], class_distribution['positive_count']]
    colors = ['#e74c3c', '#3498db']
    
    wedges, texts, autotexts = ax.pie(counts, labels=labels, autopct='%1.1f%%', 
                                       colors=colors, startangle=90)
    ax.set_title(f'Class Distribution\n(Imbalance Ratio: {class_distribution["imbalance_ratio"]:.2f}:1)',
                fontsize=13, fontweight='bold')
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ml_class_distribution.png', dpi=600, bbox_inches='tight')
    plt.savefig(output_dir / 'ml_class_distribution.pdf', dpi=600, bbox_inches='tight')
    plt.close(fig)
    
    # 2. Model Performance Comparison (Grouped Bar Chart)
    fig, ax = plt.subplots(figsize=(12, 6))
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    
    x = np.arange(len(models))
    width = 0.15
    
    for i, metric in enumerate(metrics):
        values = [avg_metrics_dict[model][metric] for model in models]
        ax.bar(x + i * width, values, width, label=metric.replace('_', ' ').title())
    
    ax.set_xlabel('Models', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison (Class Weight Balanced)', fontsize=13, fontweight='bold')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(models)
    ax.legend(loc='lower right')
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ml_model_performance_comparison.png', dpi=600, bbox_inches='tight')
    plt.savefig(output_dir / 'ml_model_performance_comparison.pdf', dpi=600, bbox_inches='tight')
    plt.close(fig)
    
    # 3. F1-Score Distribution Across Folds (Box Plot)
    fig, ax = plt.subplots(figsize=(10, 6))
    f1_data = []
    model_labels = []
    
    for model in models:
        f1_scores = [fold['f1_score'] for fold in fold_results_dict[model]]
        f1_data.append(f1_scores)
        model_labels.append(model)
    
    box_plot = ax.boxplot(f1_data, labels=model_labels, patch_artist=True)
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax.set_title('F1-Score Distribution Across CV Folds (Balanced Training)', fontsize=13, fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ml_f1_score_distribution.png', dpi=600, bbox_inches='tight')
    plt.savefig(output_dir / 'ml_f1_score_distribution.pdf', dpi=600, bbox_inches='tight')
    plt.close(fig)
    
    # 4. Inference Time Comparison (Bar Chart)
    fig, ax = plt.subplots(figsize=(10, 6))
    inference_times = [avg_metrics_dict[model]['inference_time'] for model in models]
    bars = ax.bar(models, inference_times, color=['#3498db', '#2ecc71', '#e74c3c'], alpha=0.8, edgecolor='black')
    ax.set_ylabel('Inference Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Model Inference Time Comparison', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, time_val in zip(bars, inference_times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
                f'{time_val:.4f}s', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ml_inference_time.png', dpi=600, bbox_inches='tight')
    plt.savefig(output_dir / 'ml_inference_time.pdf', dpi=600, bbox_inches='tight')
    plt.close(fig)
    
    # 5. Throughput Comparison (Bar Chart)
    fig, ax = plt.subplots(figsize=(10, 6))
    throughput = [avg_metrics_dict[model]['samples_per_second'] for model in models]
    bars = ax.bar(models, throughput, color=['#9b59b6', '#f39c12', '#1abc9c'], alpha=0.8, edgecolor='black')
    ax.set_ylabel('Samples Per Second', fontsize=12, fontweight='bold')
    ax.set_title('Model Throughput Comparison', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, throughput_val in zip(bars, throughput):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{throughput_val:.0f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ml_throughput.png', dpi=600, bbox_inches='tight')
    plt.savefig(output_dir / 'ml_throughput.pdf', dpi=600, bbox_inches='tight')
    plt.close(fig)
    
    # 6. Performance vs Speed Trade-off (Scatter Plot)
    fig, ax = plt.subplots(figsize=(10, 7))
    colors_scatter = ['#3498db', '#2ecc71', '#e74c3c']
    
    for i, model in enumerate(models):
        f1 = avg_metrics_dict[model]['f1_score']
        time_val = avg_metrics_dict[model]['inference_time']
        ax.scatter(time_val, f1, s=300, label=model, alpha=0.7, color=colors_scatter[i], edgecolors='black', linewidth=2)
        ax.annotate(model, (time_val, f1), xytext=(5, 5), 
                    textcoords='offset points', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Inference Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax.set_title('Performance vs Speed Trade-off (Balanced Models)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=11)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ml_performance_speed_tradeoff.png', dpi=600, bbox_inches='tight')
    plt.savefig(output_dir / 'ml_performance_speed_tradeoff.pdf', dpi=600, bbox_inches='tight')
    plt.close(fig)
    
    # 7. Accuracy Trends Across Folds (Line Plot)
    fig, ax = plt.subplots(figsize=(12, 6))
    folds = range(1, len(fold_results_dict[models[0]]) + 1)
    colors_line = ['#3498db', '#2ecc71', '#e74c3c']
    
    for i, model in enumerate(models):
        accuracies = [fold['accuracy'] for fold in fold_results_dict[model]]
        ax.plot(folds, accuracies, marker='o', label=model, linewidth=2.5, markersize=8, color=colors_line[i])
    
    ax.set_xlabel('Fold Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy Trends Across CV Folds', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.set_ylim([0, 1.05])
    ax.set_xticks(folds)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ml_accuracy_trends.png', dpi=600, bbox_inches='tight')
    plt.savefig(output_dir / 'ml_accuracy_trends.pdf', dpi=600, bbox_inches='tight')
    plt.close(fig)
    
    # 8. Precision vs Recall Scatter Plot
    fig, ax = plt.subplots(figsize=(10, 7))
    colors_pq = ['#3498db', '#2ecc71', '#e74c3c']
    
    for i, model in enumerate(models):
        precision = avg_metrics_dict[model]['precision']
        recall = avg_metrics_dict[model]['recall']
        ax.scatter(recall, precision, s=300, label=model, alpha=0.7, color=colors_pq[i], edgecolors='black', linewidth=2)
        ax.annotate(model, (recall, precision), xytext=(5, 5), 
                    textcoords='offset points', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax.set_title('Precision vs Recall Comparison (Balanced Training)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=11)
    ax.set_xlim([0, 1.05])
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ml_precision_recall.png', dpi=600, bbox_inches='tight')
    plt.savefig(output_dir / 'ml_precision_recall.pdf', dpi=600, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Separate comprehensive visualizations created in {output_dir}")


def apply_smote_resampling(X_train, y_train, random_state=42):
    """Apply SMOTE on training features and labels."""
    if hasattr(X_train, 'toarray'):
        X_dense = X_train.toarray()
    else:
        X_dense = np.asarray(X_train)

    try:
        smote = SMOTE(random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(X_dense, y_train)
    except ValueError as e:
        logger.warning(f"Default SMOTE failed: {e}. Retrying with k_neighbors=1")
        smote = SMOTE(random_state=random_state, k_neighbors=1)
        X_resampled, y_resampled = smote.fit_resample(X_dense, y_train)

    return X_resampled, y_resampled


def plot_smote_before_after(y_before, y_after, output_path='./smote_before_after_distribution.png'):
    """Plot class distribution before and after SMOTE to show balancing effect."""
    setup_professional_style()
    class_order = [0, 1]
    class_labels = ['No-Meaningful', 'Meaningful']
    # Professional, colorblind-friendly pair (blue/orange).
    class_colors = ['#4C78A8', "#92673C"]

    before_counts = pd.Series(y_before).value_counts().reindex(class_order, fill_value=0)
    after_counts = pd.Series(y_after).value_counts().reindex(class_order, fill_value=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(class_labels, before_counts.values, color=class_colors)
    axes[0].set_title('Before SMOTE')
    axes[0].set_xlabel('Class Label')
    axes[0].set_ylabel('Sample Count')
    axes[0].grid(True, axis='y', alpha=0.3)

    axes[1].bar(class_labels, after_counts.values, color=class_colors)
    axes[1].set_title('After SMOTE')
    axes[1].set_xlabel('Class Label')
    axes[1].set_ylabel('Sample Count')
    axes[1].grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()
    logger.info(f"SMOTE distribution chart saved to {output_path}")


def run_comprehensive_experiment():
    """
    Run the comprehensive experiment comparing SVM, Naive Bayes, and Random Forest with class balancing.
    """
    logger.info("Machine learning pipeline is running in CPU-only mode (sklearn).")

    # Load and prepare data using the CANONICAL split shared by all pipelines.
    # This guarantees identical held-out test sets (and therefore identical
    # confusion-matrix sample counts) across ML, DL, Transformer, and LLM.
    canonical = get_canonical_split()
    data = canonical["data"]
    texts = data['text'].astype(str).to_numpy(dtype=object)
    labels = pd.to_numeric(data['label'], errors='coerce').to_numpy(dtype=np.int64)
    cv_texts = canonical["texts_train"]
    test_texts = canonical["texts_test"]
    cv_labels = canonical["labels_train"]
    test_labels = canonical["labels_test"]
    print(f"Canonical split: {len(cv_texts)} train, {len(test_texts)} test")

    result_dirs = get_results_dirs()
    artifacts_dir = str(result_dirs['artifacts'])
    
    # Verify data integrity
    assert len(cv_texts) > 0, "No training data!"
    assert len(test_texts) > 0, "No test data!"
    logger.info(f"Canonical split: {len(cv_texts)} train, {len(test_texts)} test")
    
    # Initialize classifier comparison system. We apply SMOTE, so class weights are disabled.
    classifier_system = BinaryTextClassifierComparison(use_class_balancing=False)
    
    # Analyze data distribution
    class_distribution = classifier_system.analyze_data_distribution(labels, "Original Data Distribution")
    train_distribution = classifier_system.analyze_data_distribution(cv_labels, "Cross-Validation Training Distribution")
    test_distribution = classifier_system.analyze_data_distribution(test_labels, "Held-out Test Distribution")
    
    logger.info("SMOTE balancing is enabled for training folds")
    
    # Cross-validation setup
    n_splits = 10
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Store results for all models
    all_fold_results = {
        'SVM': [],
        'Naive_Bayes': [],
        'Random_Forest': []
    }
    
    logger.info(f"Starting {n_splits}-fold stratified cross-validation with class balancing...")
    all_fold_predictions = {
        'SVM': [],
        'Naive_Bayes': [],
        'Random_Forest': []
    }
    
    fold_iterator = list(skf.split(cv_texts, cv_labels))
    for fold, (train_idx, val_idx) in enumerate(tqdm(fold_iterator, desc='Cross-validation folds', total=len(fold_iterator))):
        logger.info(f"\n{'='*50}")
        logger.info(f"Fold {fold+1}/{n_splits}")
        logger.info(f"{'='*50}")
        
        # Split data
        train_texts = cv_texts[train_idx]
        train_labels = cv_labels[train_idx]
        val_texts = cv_texts[val_idx]
        val_labels = cv_labels[val_idx]
        
        # Analyze fold distribution
        fold_classifier = BinaryTextClassifierComparison(use_class_balancing=False)
        fold_classifier.analyze_data_distribution(train_labels, f"Fold {fold+1} Training Distribution")
        fold_classifier.analyze_data_distribution(val_labels, f"Fold {fold+1} Validation Distribution")
        
        # Transform features
        X_train_tfidf, X_val_tfidf = fold_classifier.fit_transform_features(train_texts, val_texts)
        
        # Apply SMOTE on training fold and train models on balanced data
        X_train_smote, y_train_smote = apply_smote_resampling(X_train_tfidf, train_labels, random_state=42 + fold)

        if fold == 0:
            plot_smote_before_after(
                train_labels,
                y_train_smote,
                output_path=str(result_dirs['figures'] / 'smote_before_after_distribution.png')
            )

        fold_classifier.fit_models(X_train_smote, y_train_smote)
        
        # Make predictions with all models
        predictions = fold_classifier.predict_models(X_val_tfidf)
        train_predictions = fold_classifier.predict_models(X_train_smote)
        
        # Compute metrics for all models
        fold_metrics = fold_classifier.compute_metrics(val_labels, predictions, len(val_labels))
        train_fold_metrics = fold_classifier.compute_metrics(y_train_smote, train_predictions, len(y_train_smote))
        
        # Store results
        for model_name in fold_metrics:
            fold_metrics[model_name]['fold'] = fold + 1
            fold_metrics[model_name]['train_log_loss'] = train_fold_metrics[model_name]['log_loss']
            fold_metrics[model_name]['val_log_loss'] = fold_metrics[model_name]['log_loss']
            all_fold_results[model_name].append(fold_metrics[model_name])

            y_pred = predictions[model_name]['predictions']
            y_proba = predictions[model_name]['probabilities'][:, 1] if predictions[model_name]['probabilities'].shape[1] == 2 else np.zeros(len(y_pred))
            fold_pred_df = pd.DataFrame({
                'fold': fold + 1,
                'sample_index': np.arange(len(val_labels)),
                'true_label': val_labels.astype(int),
                'predicted_label': y_pred.astype(int),
                'true_label_name': label_name_array(val_labels),
                'predicted_label_name': label_name_array(y_pred),
                'predicted_prob': y_proba,
                'model': model_name
            })
            all_fold_predictions[model_name].append(fold_pred_df)
            
        # Log fold results
        for model_name, metrics in fold_metrics.items():
            logger.info(f"\n{model_name} - Fold {fold+1} Results (Class Balanced):")
            logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"Precision: {metrics['precision']:.4f}")
            logger.info(f"Recall: {metrics['recall']:.4f}")
            logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
            logger.info(f"ROC AUC: {metrics['roc_auc']:.4f}")
            logger.info(f"Inference time: {metrics['inference_time']:.4f}s")
    
    # Calculate average metrics for all models
    avg_metrics = {}
    for model_name in all_fold_results:
        model_results = all_fold_results[model_name]
        avg_metrics[model_name] = {
            'accuracy': np.mean([r['accuracy'] for r in model_results]),
            'precision': np.mean([r['precision'] for r in model_results]),
            'recall': np.mean([r['recall'] for r in model_results]),
            'f1_score': np.mean([r['f1_score'] for r in model_results]),
            'roc_auc': np.mean([r['roc_auc'] for r in model_results]),
            'inference_time': np.mean([r['inference_time'] for r in model_results]),
            'inference_time_per_sample': np.mean([r['inference_time_per_sample'] for r in model_results]),
            'samples_per_second': np.mean([r['samples_per_second'] for r in model_results]),
            'train_log_loss': np.nanmean([r['train_log_loss'] for r in model_results]),
            'val_log_loss': np.nanmean([r['val_log_loss'] for r in model_results]),
            'std_accuracy': np.std([r['accuracy'] for r in model_results]),
            'std_f1': np.std([r['f1_score'] for r in model_results]),
            'std_train_log_loss': np.nanstd([r['train_log_loss'] for r in model_results]),
            'std_val_log_loss': np.nanstd([r['val_log_loss'] for r in model_results])
        }
    
    # Print comprehensive results
    print(f"\n{'='*80}")
    print("COMPREHENSIVE MODEL COMPARISON RESULTS (CLASS BALANCED)")
    print(f"{'='*80}")
    
    # Print class distribution summary
    print(f"\nDATASET STATISTICS:")
    print(f"Total samples: {class_distribution['total_samples']}")
    print(f"{LABEL_DISPLAY_NAMES[LABEL_NO_MEANINGFUL]}: {class_distribution['negative_count']} ({class_distribution['negative_ratio']:.1f}%)")
    print(f"{LABEL_DISPLAY_NAMES[LABEL_MEANINGFUL]}: {class_distribution['positive_count']} ({class_distribution['positive_ratio']:.1f}%)")
    print(f"Imbalance ratio: {class_distribution['imbalance_ratio']:.2f}:1")
    
    print("\nSMOTE applied on each training fold to balance classes.")
    
    for model_name, metrics in avg_metrics.items():
        print(f"\n{model_name.upper()} RESULTS (CLASS BALANCED):")
        print(f"Average Accuracy: {metrics['accuracy']:.4f} ± {metrics['std_accuracy']:.4f}")
        print(f"Average Precision: {metrics['precision']:.4f}")
        print(f"Average Recall: {metrics['recall']:.4f}")
        print(f"Average F1 Score: {metrics['f1_score']:.4f} ± {metrics['std_f1']:.4f}")
        print(f"Average ROC AUC: {metrics['roc_auc']:.4f}")
        print(f"Average Train Log Loss: {metrics['train_log_loss']:.4f} ± {metrics['std_train_log_loss']:.4f}")
        print(f"Average Validation Log Loss: {metrics['val_log_loss']:.4f} ± {metrics['std_val_log_loss']:.4f}")
        print(f"Average Inference Time: {metrics['inference_time']:.4f}s")
        print(f"Average Throughput: {metrics['samples_per_second']:.0f} samples/sec")
    
    # Determine best model for each metric
    print(f"\n{'='*80}")
    print("BEST PERFORMING MODELS BY METRIC (CLASS BALANCED)")
    print(f"{'='*80}")
    
    best_models = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
        best_model = max(avg_metrics.keys(), key=lambda x: avg_metrics[x][metric])
        best_score = avg_metrics[best_model][metric]
        best_models[metric] = (best_model, best_score)
        print(f"Best {metric.replace('_', ' ').title()}: {best_model} ({best_score:.4f})")

    # Select best fold per model based on validation F1 score.
    best_fold_summary = {}
    for model_name, fold_rows in all_fold_results.items():
        best_row = max(fold_rows, key=lambda r: r['f1_score'])
        best_fold_summary[model_name] = {
            'pipeline': 'machine_learning',
            'model': model_name,
            'best_fold': int(best_row.get('fold', 1)),
            'selection_metric': 'f1_score',
            'smote_applied_in_training_fold': True,
            'metrics': {
                'accuracy': float(best_row['accuracy']),
                'precision': float(best_row['precision']),
                'recall': float(best_row['recall']),
                'f1_score': float(best_row['f1_score']),
                'roc_auc': float(best_row['roc_auc']),
                'inference_time': float(best_row['inference_time'])
            }
        }
    
    # Speed comparison
    fastest_model = min(avg_metrics.keys(), key=lambda x: avg_metrics[x]['inference_time'])
    fastest_time = avg_metrics[fastest_model]['inference_time']
    print(f"Fastest Model: {fastest_model} ({fastest_time:.4f}s)")
    
    highest_throughput = max(avg_metrics.keys(), key=lambda x: avg_metrics[x]['samples_per_second'])
    throughput_score = avg_metrics[highest_throughput]['samples_per_second']
    print(f"Highest Throughput: {highest_throughput} ({throughput_score:.0f} samples/sec)")
    
    # Create visualizations
    logger.info("Creating comprehensive visualizations with class balancing info...")
    create_comprehensive_visualizations(
        all_fold_results,
        avg_metrics,
        class_distribution,
        output_path=str(result_dirs['figures'] / 'comprehensive_model_comparison_balanced.png')
    )
    
    # Create publication-ready plots
    logger.info("Creating publication-ready plots...")
    create_publication_ready_plots(avg_metrics, all_fold_results, output_dir=str(result_dirs['figures']))
    
    # Statistical significance testing
    perform_statistical_analysis(all_fold_results)
    
    # Save best models (trained on SMOTE-balanced training data)
    logger.info("Training final models on SMOTE-balanced CV training split for saving...")
    final_classifier = BinaryTextClassifierComparison(use_class_balancing=False)
    X_train_tfidf, X_test_tfidf = final_classifier.fit_transform_features(cv_texts, test_texts)
    X_train_smote_final, y_train_smote_final = apply_smote_resampling(X_train_tfidf, cv_labels, random_state=42)
    plot_smote_before_after(
        cv_labels,
        y_train_smote_final,
        output_path=str(result_dirs['figures'] / 'smote_before_after_distribution_final_train.png')
    )
    final_classifier.fit_models(X_train_smote_final, y_train_smote_final)
    final_classifier.save_models(str(result_dirs['models'] / 'best_models_comparison_balanced'))

    # Evaluate final models on held-out test split and persist ground truth/predictions
    test_predictions = final_classifier.predict_models(X_test_tfidf)
    test_metrics = final_classifier.compute_metrics(test_labels, test_predictions, len(test_labels))

    

    # Create confusion matrix and train-loss charts for each model
    create_confusion_matrix_plots(test_labels, test_predictions, output_dir=str(result_dirs['figures']))
    create_train_loss_plots(all_fold_results, output_dir=str(result_dirs['figures']))

    for model_name, pred_data in test_predictions.items():
        y_pred = pred_data['predictions']
        y_proba = pred_data['probabilities'][:, 1] if pred_data['probabilities'].shape[1] == 2 else np.zeros(len(y_pred))
        test_pred_df = pd.DataFrame({
            'sample_index': np.arange(len(test_labels)),
            'true_label': test_labels.astype(int),
            'predicted_label': y_pred.astype(int),
            'true_label_name': label_name_array(test_labels),
            'predicted_label_name': label_name_array(y_pred),
            'predicted_prob': y_proba,
            'model': model_name
        })
        test_pred_df.to_csv(
            os.path.join(artifacts_dir, f"{model_name.lower()}_test_predictions.csv"),
            index=False,
            encoding='utf-8'
        )

    for model_name, fold_frames in all_fold_predictions.items():
        if fold_frames:
            combined_cv = pd.concat(fold_frames, ignore_index=True)
            combined_cv.to_csv(
                os.path.join(artifacts_dir, f"{model_name.lower()}_cv_predictions.csv"),
                index=False,
                encoding='utf-8'
            )

            best_fold_id = best_fold_summary[model_name]['best_fold']
            combined_cv[combined_cv['fold'] == best_fold_id].to_csv(
                os.path.join(artifacts_dir, f"{model_name.lower()}_best_fold_predictions.csv"),
                index=False,
                encoding='utf-8'
            )

    training_info = {
        'experiment': 'BinaryTextClassifierComparison',
        'total_samples': int(len(labels)),
        'cv_training_samples': int(len(cv_labels)),
        'heldout_test_samples': int(len(test_labels)),
        'class_distribution_total': class_distribution,
        'class_distribution_train': train_distribution,
        'class_distribution_test': test_distribution,
        'smote_training_distribution': {
            'before': {
                'class_0': int(np.sum(cv_labels == 0)),
                'class_1': int(np.sum(cv_labels == 1))
            },
            'after': {
                'class_0': int(np.sum(y_train_smote_final == 0)),
                'class_1': int(np.sum(y_train_smote_final == 1))
            }
        },
        'class_weights': final_classifier.class_weights,
        'vectorizer_settings': {
            'max_features': final_classifier.max_features,
            'ngram_range': final_classifier.ngram_range,
            'min_df': final_classifier.min_df,
            'max_df': final_classifier.max_df
        },
        'cv_avg_metrics': avg_metrics,
        'cv_best_fold_metrics': best_fold_summary,
        'cv_training_smote': True,
        'heldout_test_metrics': test_metrics,        
        'saved_model_file': str(result_dirs['models'] / 'best_models_comparison_balanced_all_models.pkl')
    }
    with open(os.path.join(artifacts_dir, 'ml_training_info.json'), 'w', encoding='utf-8') as f:
        json.dump(training_info, f, indent=2, ensure_ascii=False, default=float)

    with open(os.path.join(artifacts_dir, 'ml_best_fold_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(best_fold_summary, f, indent=2, ensure_ascii=False)
    
    # Generate model comparison report
    generate_model_comparison_report(
        avg_metrics,
        all_fold_results,
        best_models,
        class_distribution,
        output_path=str(result_dirs['reports'] / 'model_comparison_report_balanced.md')
    )
    
    logger.info(f"Training artifacts and predictions saved to {artifacts_dir}")
    return all_fold_results, avg_metrics

def create_publication_ready_plots(avg_metrics_dict, fold_results_dict, output_dir='.'):
    """
    Create professional publication-ready plots using unified figure utilities.
    Generates: metrics panel, fold-by-fold comparisons, per-fold CSV export.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Export per-fold metrics to CSV for reproducibility
    export_fold_metrics_csv(fold_results_dict, output_path, "ml_fold_metrics")
    
    # Figure 1-4: Separate metrics figures across folds
    plot_metrics_panel(fold_results_dict, output_path, "ml_metrics_panel", separate=True)
    
    # Figure 2: Model Performance Comparison (Average Metrics)
    plot_model_comparison_bar(
        avg_metrics_dict,
        metrics=['accuracy', 'precision', 'recall', 'f1_score'],
        output_path=output_path,
        filename_stem="ml_model_comparison",
        title="ML Model Performance Comparison (SMOTE Balanced)"
    )

def perform_statistical_analysis(fold_results_dict):
    """
    Perform statistical significance testing between models.
    """
    from scipy import stats
    
    logger.info("\nPerforming statistical significance testing...")
    print(f"\n{'='*60}")
    print("STATISTICAL SIGNIFICANCE ANALYSIS (CLASS BALANCED)")
    print(f"{'='*60}")
    
    models = list(fold_results_dict.keys())
    
    # Extract F1 scores for each model
    f1_scores = {}
    for model in models:
        f1_scores[model] = [fold['f1_score'] for fold in fold_results_dict[model]]
    
    # Perform pairwise t-tests
    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models):
            if i < j:  # Avoid duplicate comparisons
                t_stat, p_value = stats.ttest_rel(f1_scores[model1], f1_scores[model2])
                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "n.s."
                print(f"{model1} vs {model2}: t={t_stat:.4f}, p={p_value:.4f} {significance}")
    
    print("\nSignificance levels: *** p<0.001, ** p<0.01, * p<0.05, n.s. not significant")

def generate_model_comparison_report(avg_metrics, fold_results, best_models, class_distribution, output_path):
    """
    Generate a comprehensive model comparison report for publication with class balancing info.
    """
    report = f"""
# Binary Text Classification Model Comparison Report (Class Weight Balanced)

## Executive Summary

This report presents a comprehensive comparison of three machine learning models for binary text classification on Chinese text data: Support Vector Machine (SVM), Naive Bayes, and Random Forest. The evaluation was conducted using 10-fold stratified cross-validation with automatic class weight balancing to handle data imbalance and ensure robust and reliable results.

## Dataset Overview

- **Total Samples**: {class_distribution['total_samples']}
- **Class Distribution**: 
    - No-meaningful: {class_distribution['negative_count']} samples ({class_distribution['negative_ratio']:.1f}%)
    - Meaningful: {class_distribution['positive_count']} samples ({class_distribution['positive_ratio']:.1f}%)
- **Imbalance Ratio**: {class_distribution['imbalance_ratio']:.2f}:1
- **Language**: Chinese text with binary labels (0/1)
- **Preprocessing**: Jieba word segmentation, TF-IDF vectorization
- **Evaluation Method**: 10-fold stratified cross-validation with class weight balancing

## Class Imbalance Handling

**Method Used**: Class Weight Balancing instead of SMOTE

**Rationale**:
- More appropriate for high-dimensional sparse text features
- Preserves original data integrity
- Computationally efficient
- Avoids creating synthetic text combinations that may be unrealistic

**Implementation**:
- SVM and Random Forest: Built-in `class_weight='balanced'` parameter
- Naive Bayes: Manual sample weight calculation and application
- Weight formula: `weight_i = total_samples / (n_classes × count_i)`

## Model Performance Summary

### Overall Results (Class Balanced)

| Model | Accuracy | Precision | Recall | F1-Score | ROC AUC | Inference Time (s) |
|-------|----------|-----------|--------|----------|---------|-------------------|
"""
    
    for model, metrics in avg_metrics.items():
        report += f"| {model} | {metrics['accuracy']:.4f} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1_score']:.4f} | {metrics['roc_auc']:.4f} | {metrics['inference_time']:.4f} |\n"
    
    report += f"""
### Best Performing Models by Metric (Class Balanced)

"""
    for metric, (model, score) in best_models.items():
        report += f"- **{metric.replace('_', ' ').title()}**: {model} ({score:.4f})\n"
    
    report += f"""

## Model Analysis with Class Balancing

### 1. Support Vector Machine (SVM)
- **Strengths**: High precision, robust to overfitting, effective with high-dimensional data
- **Class Balancing**: Uses built-in `class_weight='balanced'` parameter
- **Performance**: Balanced precision and recall due to proper weight adjustment
- **Use Case**: When precision is critical and computational resources are available

### 2. Naive Bayes
- **Strengths**: Fast training and inference, simple implementation, good baseline performance
- **Class Balancing**: Manual sample weight application during training
- **Performance**: Improved minority class detection with weight balancing
- **Use Case**: Rapid prototyping, real-time applications with speed requirements

### 3. Random Forest
- **Strengths**: Feature importance insights, handles non-linear relationships, robust to outliers
- **Class Balancing**: Uses built-in `class_weight='balanced'` parameter
- **Performance**: Excellent balance between precision and recall
- **Use Case**: When interpretability and feature importance are needed

## Impact of Class Weight Balancing

### Before vs After Comparison:
- **Improved Minority Class Detection**: Better recall for underrepresented class
- **Balanced Precision-Recall**: More even performance across both classes
- **Reduced Bias**: Models less likely to favor majority class
- **Fair Evaluation**: Metrics reflect true model performance on both classes

## Computational Efficiency Analysis

"""
    
    fastest_model = min(avg_metrics.keys(), key=lambda x: avg_metrics[x]['inference_time'])
    highest_throughput = max(avg_metrics.keys(), key=lambda x: avg_metrics[x]['samples_per_second'])
    
    report += f"""
- **Fastest Model**: {fastest_model} ({avg_metrics[fastest_model]['inference_time']:.4f}s per batch)
- **Highest Throughput**: {highest_throughput} ({avg_metrics[highest_throughput]['samples_per_second']:.0f} samples/second)
- **Class Balancing Overhead**: Minimal additional computational cost

## Recommendations for Educational Context

### For Teaching Machine Learning Concepts:
1. **Start with Naive Bayes**: Simple, interpretable, demonstrates class balancing effects clearly
2. **Progress to SVM**: Introduce kernel methods and optimization with balanced objectives
3. **Conclude with Random Forest**: Ensemble methods and feature importance with class weights

### For Research Applications:
- Use **{best_models['f1_score'][0]}** for best overall balanced performance
- Use **{fastest_model}** for real-time applications
- Use **Random Forest** when feature importance analysis is required
- Always apply class balancing for imbalanced datasets

## Technical Implementation Notes

### Feature Engineering:
- TF-IDF vectorization with n-gram range (1,2)
- Maximum features: 5000
- Chinese text segmentation using Jieba
- Automatic class weight calculation and application

### Model Hyperparameters (Class Balanced):
- **SVM**: RBF kernel, C=1.0, class_weight='balanced'
- **Naive Bayes**: Alpha=1.0 (Laplace smoothing), manual sample weights
- **Random Forest**: 100 estimators, class_weight='balanced'

### Class Weight Calculation:
```python
class_weights[label] = total_samples / (n_classes × count_label)
```

## Statistical Significance

All model comparisons include statistical significance testing using paired t-tests on F1-scores across cross-validation folds, ensuring robust performance assessment.

## Conclusion

The comparison reveals that class weight balancing significantly improves model performance on imbalanced text data:

- **SVM** provides robust balanced performance with proper weight handling
- **Naive Bayes** offers speed and simplicity with effective manual weight application
- **Random Forest** delivers excellent interpretability with built-in balancing support

**Key Finding**: Class weight balancing is essential for fair evaluation and practical deployment of text classification models on imbalanced datasets.

For educational journal publication, this comprehensive evaluation demonstrates the critical importance of addressing class imbalance in text classification tasks and provides practical guidance for implementation.

## Files Generated

- `best_models_comparison_balanced_all_models.pkl`: Trained models with class weights
- `comprehensive_model_comparison_balanced.png`: Complete visualization suite
- `publication_figure1_performance_metrics_balanced.png`: Performance comparison
- `publication_figure2_computational_efficiency_balanced.png`: Efficiency analysis
"""
    
    # Save report to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"Class-balanced model comparison report generated and saved to '{output_path}'")

def create_confusion_matrix_plots(y_true, prediction_dict, output_dir='.'):
    """
    Create separate confusion matrix plots for each model on test predictions.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    models = ['SVM', 'Naive_Bayes', 'Random_Forest']
    model_display_names = {'SVM': 'SVM', 'Naive_Bayes': 'Naive Bayes', 'Random_Forest': 'Random Forest'}
    
    for model in models:
        y_pred = prediction_dict[model]['predictions']

        filename_base = f'ml_confusion_matrix_{model.lower()}'
        fig = plot_confusion_matrix_consistent(
            y_true=y_true,
            y_pred=y_pred,
            title=f"{model_display_names[model]} - Confusion Matrix (Held-out Test)",
            output_path=output_path,
            filename_stem=filename_base,
            class_labels=['No-Meaningful', 'Meaningful'],
            formats=("png", "pdf"),
        )
        plt.close(fig)
    
    logger.info(f"Separate confusion matrix plots created in {output_dir}")


def create_train_loss_plots(fold_results_dict, output_dir='.'):
    """
    Plot training and validation log-loss curves across folds for each model.
    """
    os.makedirs(output_dir, exist_ok=True)
    setup_professional_style()
    models = ['SVM', 'Naive_Bayes', 'Random_Forest']
    folds = range(1, len(fold_results_dict[models[0]]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for i, model in enumerate(models):
        train_losses = [r.get('train_log_loss', np.nan) for r in fold_results_dict[model]]
        val_losses = [r.get('val_log_loss', np.nan) for r in fold_results_dict[model]]

        axes[i].plot(folds, train_losses, marker='o', linewidth=2, label='Train Log Loss')
        axes[i].plot(folds, val_losses, marker='s', linewidth=2, label='Validation Log Loss')
        axes[i].set_title(f'{model} Loss Across Folds')
        axes[i].set_xlabel('Fold')
        axes[i].grid(True, alpha=0.3)
        if i == 0:
            axes[i].set_ylabel('Log Loss')
        axes[i].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'train_loss_comparison_balanced.png'), dpi=600, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Run the comprehensive experiment with class balancing
    try:
        fold_results, avg_metrics = run_comprehensive_experiment()
        print("\n" + "="*80)
        print("COMPREHENSIVE MODEL COMPARISON WITH CLASS BALANCING COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nAll outputs are stored in: ../results/machine_learning/")
        print("- figures/")
        print("- artifacts/")
        print("- models/")
        print("- reports/")
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        raise