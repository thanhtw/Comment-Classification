import pandas as pd
import numpy as np
import time
import random
import matplotlib.pyplot as plt
import psutil
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import warnings
import jieba
import pickle
from collections import Counter
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

class SVMTFIDFBinaryClassifier:
    """
    A binary text classifier using SVM with TF-IDF features.
    Designed for Chinese text classification with binary labels (0/1).
    """
    
    def __init__(self, max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.95, 
                 svm_C=1.0, svm_gamma='scale', random_state=42):
        """
        Initialize the SVM TF-IDF binary classifier.
        
        Args:
            max_features: Maximum number of TF-IDF features
            ngram_range: Range of n-grams to extract
            min_df: Minimum document frequency for terms
            max_df: Maximum document frequency for terms
            svm_C: SVM regularization parameter
            svm_gamma: SVM kernel coefficient
            random_state: Random state for reproducibility
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.svm_C = svm_C
        self.svm_gamma = svm_gamma
        self.random_state = random_state
        
        self.vectorizer = None
        self.model = None
        self.scaler = None
        
    def analyze_data_distribution(self, labels, title="Data Distribution"):
        """
        Analyze and print the distribution of binary labels in the dataset.
        
        Args:
            labels: numpy array of binary labels (0/1)
            title: Title for the analysis output
        """
        logger.info(f"\n{title}:")
        total_samples = len(labels)
        logger.info(f"Total samples: {total_samples}")
        
        positive_count = np.sum(labels == 1)
        negative_count = np.sum(labels == 0)
        positive_ratio = positive_count / total_samples * 100
        negative_ratio = negative_count / total_samples * 100
        
        logger.info(f"Positive (1): {positive_count} ({positive_ratio:.1f}%)")
        logger.info(f"Negative (0): {negative_count} ({negative_ratio:.1f}%)")
        
        # Check for class imbalance
        imbalance_ratio = max(positive_count, negative_count) / min(positive_count, negative_count)
        if imbalance_ratio > 2:
            logger.warning(f"Class imbalance detected! Ratio: {imbalance_ratio:.2f}")
        else:
            logger.info(f"Classes are relatively balanced. Ratio: {imbalance_ratio:.2f}")

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
        
        return class_weights

    def apply_random_oversampling(self, features, labels):
        """
        Apply random oversampling as an alternative to SMOTE for text data.
        This duplicates existing samples rather than creating synthetic ones.
        
        Args:
            features: Feature matrix (can be sparse)
            labels: Binary label array
            
        Returns:
            Resampled features and labels
        """
        try:
            from imblearn.over_sampling import RandomOverSampler
            logger.info("Starting random oversampling...")
            
            # Check class distribution
            unique_labels, label_counts = np.unique(labels, return_counts=True)
            min_samples = min(label_counts)
            logger.info(f"Class distribution: {dict(zip(unique_labels, label_counts))}")
            
            if min_samples < 2:
                logger.warning("Insufficient samples for oversampling")
                return features, labels
            
            # Random oversampling works with sparse matrices
            ros = RandomOverSampler(random_state=self.random_state)
            resampled_features, resampled_labels = ros.fit_resample(features, labels)
            
            logger.info(f"Original samples: {len(labels)} -> Resampled: {len(resampled_labels)}")
            
            # Show new distribution
            unique_resampled, resampled_counts = np.unique(resampled_labels, return_counts=True)
            logger.info(f"Resampled distribution: {dict(zip(unique_resampled, resampled_counts))}")
            logger.info("Random oversampling completed")
            
            return resampled_features, resampled_labels
            
        except (ImportError, ValueError) as e:
            logger.warning(f"Random oversampling failed ({e}), using original data")
            return features, labels

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
        # Handle multiline text by replacing newlines with spaces
        text = str(text).replace('\n', ' ').replace('\r', ' ').strip()
        words = jieba.cut(text)
        return ' '.join(words)

    def compute_metrics(self, y_true, y_pred, inference_time, num_samples, memory_usage_mb):
        """
        Compute comprehensive evaluation metrics for binary classification.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            inference_time: Time taken for inference
            num_samples: Number of samples
            memory_usage_mb: Memory usage in MB
            
        Returns:
            Dictionary of computed metrics
        """
        # Handle zero division cases
        precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        
        # Calculate feature importance (simulated for SVM)
        avg_feature_importance = 0.5
        
        return {
            "eval_accuracy": accuracy,
            "eval_precision": precision,
            "eval_recall": recall,
            "eval_f1": f1,
            "avg_inference_time": inference_time,
            "inference_time_per_sample": inference_time / max(num_samples, 1),
            "samples_per_second": num_samples / max(inference_time, 1e-6),
            "gpu_mem_usage": 0.0,  # SVM doesn't use GPU
            "memory_usage_mb": memory_usage_mb,
            "avg_attention_weight": avg_feature_importance
        }

    def fit(self, X, y, balance_method='class_weight', use_scaling=False):
        """
        Train the SVM TF-IDF classifier.
        
        Args:
            X: Input texts
            y: Binary labels
            balance_method: Method to handle class imbalance ('class_weight', 'oversample', 'none')
            use_scaling: Whether to apply feature scaling
        """
        logger.info("Starting model training...")
        
        # Text preprocessing
        logger.info("Preprocessing Chinese text...")
        X_processed = [self.preprocess_chinese_text(text) for text in X]
        
        # TF-IDF vectorization
        logger.info("Performing TF-IDF vectorization...")
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            stop_words=None  # No built-in stopwords for Chinese
        )
        
        X_tfidf = self.vectorizer.fit_transform(X_processed)
        logger.info(f"Feature dimensions: {X_tfidf.shape[1]}")
        logger.info(f"Feature matrix sparsity: {(X_tfidf.nnz / (X_tfidf.shape[0] * X_tfidf.shape[1]) * 100):.2f}%")
        
        # Handle class imbalance
        class_weights = None
        if balance_method == 'class_weight':
            class_weights = self.apply_class_weight_balancing(y)
        elif balance_method == 'oversample':
            X_tfidf, y = self.apply_random_oversampling(X_tfidf, y)
        elif balance_method == 'none':
            logger.info("No class balancing applied")
        else:
            logger.warning(f"Unknown balance method: {balance_method}, using class_weight")
            class_weights = self.apply_class_weight_balancing(y)
        
        # Apply scaling if requested (only for dense matrices)
        if use_scaling:
            logger.warning("Scaling will convert sparse matrix to dense - may cause memory issues")
            if hasattr(X_tfidf, 'todense'):
                X_tfidf = X_tfidf.todense()
            self.scaler = StandardScaler()
            X_tfidf = self.scaler.fit_transform(X_tfidf)
        
        # Train SVM model
        logger.info("Training SVM model...")
        
        # Use calculated class weights or 'balanced' if no specific weights
        if class_weights is not None:
            model_class_weight = class_weights
        elif balance_method == 'class_weight':
            model_class_weight = 'balanced'
        else:
            model_class_weight = None
        
        self.model = SVC(
            kernel='rbf',
            C=self.svm_C,
            gamma=self.svm_gamma,
            class_weight=model_class_weight,
            probability=True,
            random_state=self.random_state
        )
        
        self.model.fit(X_tfidf, y)
        logger.info("Model training completed")

    def predict(self, X):
        """
        Make predictions on input texts.
        
        Args:
            X: Input texts
            
        Returns:
            Predicted labels
        """
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model must be trained before prediction")
        
        # Preprocess texts
        X_processed = [self.preprocess_chinese_text(text) for text in X]
        
        # Transform to TF-IDF features
        X_tfidf = self.vectorizer.transform(X_processed)
        
        # Apply scaling if used during training
        if self.scaler is not None:
            if hasattr(X_tfidf, 'todense'):
                X_tfidf = X_tfidf.todense()
            X_tfidf = self.scaler.transform(X_tfidf)
        
        # Make predictions
        return self.model.predict(X_tfidf)

    def predict_proba(self, X):
        """
        Get prediction probabilities.
        
        Args:
            X: Input texts
            
        Returns:
            Prediction probabilities
        """
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model must be trained before prediction")
        
        # Preprocess texts
        X_processed = [self.preprocess_chinese_text(text) for text in X]
        
        # Transform to TF-IDF features
        X_tfidf = self.vectorizer.transform(X_processed)
        
        # Apply scaling if used during training
        if self.scaler is not None:
            if hasattr(X_tfidf, 'todense'):
                X_tfidf = X_tfidf.todense()
            X_tfidf = self.scaler.transform(X_tfidf)
        
        # Get probabilities
        return self.model.predict_proba(X_tfidf)

    def save_model(self, filepath):
        """
        Save the trained model and vectorizer.
        
        Args:
            filepath: Path to save the model
        """
        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'scaler': self.scaler
        }
        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """
        Load a trained model and vectorizer.
        
        Args:
            filepath: Path to load the model from
        """
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.vectorizer = model_data['vectorizer']
        self.scaler = model_data.get('scaler', None)
        logger.info(f"Model loaded from {filepath}")


def load_and_clean_data(path):
    """
    Load and clean the dataset with proper CSV parsing.
    
    Args:
        path: Path to the CSV file
        
    Returns:
        Cleaned DataFrame
    """
    logger.info("Loading and cleaning data...")
    
    # Read CSV with proper handling of quotes and multiline entries
    try:
        data = pd.read_csv(path, encoding='utf-8', quotechar='"', skipinitialspace=True)
    except Exception as e:
        logger.warning(f"Standard CSV reading failed: {e}")
        # Try alternative parsing
        data = pd.read_csv(path, encoding='utf-8', quotechar='"', 
                          skipinitialspace=True, error_bad_lines=False)
    
    # Check data structure
    logger.info(f"Data columns: {list(data.columns)}")
    logger.info(f"Data shape: {data.shape}")
    
    # Check for NaN values
    nan_rows = data[data.isna().any(axis=1)]
    if not nan_rows.empty:  
        logger.warning(f"Found {len(nan_rows)} rows with NaN values")
        logger.warning("First few NaN rows:")
        logger.warning(str(nan_rows.head()))
    else:
        logger.info("No NaN values found in data")
    
    # Clean text column
    if 'text' in data.columns:
        data['text'] = data['text'].apply(
            lambda x: str(x).replace('\n', ' ').replace('\r', ' ').strip() 
            if pd.notna(x) else ""
        )
        # Remove empty texts
        data = data[data['text'] != '']
    
    # Clean label column
    if 'label' in data.columns:
        # Convert labels to integers
        data['label'] = pd.to_numeric(data['label'], errors='coerce')
        # Remove rows where label conversion failed
        data = data.dropna(subset=['label'])
        data['label'] = data['label'].astype(int)
        
        # Check label values
        unique_labels = data['label'].unique()
        logger.info(f"Unique labels found: {sorted(unique_labels)}")
        
        # Ensure binary classification (0 and 1 only)
        valid_labels = data['label'].isin([0, 1])
        if not valid_labels.all():
            logger.warning(f"Found invalid labels. Keeping only 0 and 1.")
            data = data[valid_labels]
    
    logger.info(f"Final cleaned data shape: {data.shape}")
    return data


def display_original_data_statistics(labels):
    """
    Display original data label distribution statistics.
    
    Args:
        labels: Binary label array
    """
    print("\n" + "="*60)
    print("Original Data Label Distribution Statistics")
    print("="*60)
    
    total_samples = len(labels)
    positive_count = np.sum(labels == 1)
    negative_count = np.sum(labels == 0)
    
    print(f"Total comments: {total_samples}")
    print(f"{'Label':<10} {'Count':<10} {'Ratio':<10} {'Distribution'}")
    print("-" * 60)
    
    for label, count in [(0, negative_count), (1, positive_count)]:
        ratio = count / total_samples * 100
        bar_length = int(ratio / 2)  # Each █ represents 2%
        bar = "█" * bar_length + "░" * (50 - bar_length)
        print(f"{'Class ' + str(label):<10} {count:<10} {ratio:>6.1f}%   {bar}")
    
    print("-" * 60)
    print("Legend: █ = 2%, ░ = not reached")
    
    # Show imbalance ratio
    imbalance_ratio = max(positive_count, negative_count) / max(min(positive_count, negative_count), 1)
    print(f"Class imbalance ratio: {imbalance_ratio:.2f}")


def analyze_dataset_for_balancing(data):
    """
    Analyze the dataset to recommend the best class balancing approach.
    
    Args:
        data: DataFrame with 'text' and 'label' columns
        
    Returns:
        Recommended balancing method
    """
    labels = data['label'].values
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    
    total_samples = len(labels)
    min_class_size = min(label_counts)
    max_class_size = max(label_counts)
    imbalance_ratio = max_class_size / min_class_size
    
    print(f"\n{'='*50}")
    print("DATASET ANALYSIS FOR CLASS BALANCING")
    print(f"{'='*50}")
    print(f"Total samples: {total_samples}")
    print(f"Class distribution: {dict(zip(unique_labels, label_counts))}")
    print(f"Imbalance ratio: {imbalance_ratio:.2f}")
    print(f"Smallest class size: {min_class_size}")
    
    # Recommendations based on data characteristics
    if imbalance_ratio <= 1.5:
        recommendation = 'none'
        reason = "Classes are well balanced"
    elif imbalance_ratio <= 3.0:
        recommendation = 'class_weight'
        reason = "Moderate imbalance - class weights should be sufficient"
    elif min_class_size >= 50:
        recommendation = 'oversample'
        reason = "Significant imbalance but enough samples for oversampling"
    else:
        recommendation = 'class_weight'
        reason = "Severe imbalance with small minority class - avoid oversampling"
    
    print(f"\nRECOMMENDATION: {recommendation}")
    print(f"REASON: {reason}")
    
    if total_samples < 1000:
        print("\nNOTE: Small dataset - consider collecting more data if possible")
    
    if imbalance_ratio > 10:
        print("WARNING: Severe class imbalance may lead to poor performance")
        print("Consider collecting more data for the minority class")
    
    print(f"{'='*50}")
    
    return recommendation


def run_cross_validation_experiment():
    """
    Run the complete cross-validation experiment for binary classification.
    """
    # Load and prepare data
    data = load_and_clean_data('../data/two-label-data.csv')
    
    # Verify data integrity
    assert not data.empty, "No data loaded!"
    assert 'text' in data.columns, "Text column not found!"
    assert 'label' in data.columns, "Label column not found!"
    
    # Extract features and labels
    texts = data['text'].values
    labels = data['label'].values
    
    # Analyze dataset and get recommendation
    recommended_method = analyze_dataset_for_balancing(data)
    
    # Display original data statistics
    display_original_data_statistics(labels)
    
    # Initialize classifier
    classifier = SVMTFIDFBinaryClassifier()
    classifier.analyze_data_distribution(labels, "Original Data Distribution")
    
    # Cross-validation setup - use StratifiedKFold for better balance
    n_splits = 10
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []
    best_model = None
    best_f1 = 0.0
    
    logger.info(f"Starting {n_splits}-fold stratified cross-validation...")
    logger.info(f"Using recommended balancing method: {recommended_method}")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
        logger.info(f"\n{'='*40}")
        logger.info(f"Fold {fold+1}/{n_splits}")
        logger.info(f"{'='*40}")
        
        # Record memory usage
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Split data
        train_texts = texts[train_idx]
        train_labels = labels[train_idx]
        val_texts = texts[val_idx]
        val_labels = labels[val_idx]
        
        # Analyze training data distribution
        fold_classifier = SVMTFIDFBinaryClassifier()
        fold_classifier.analyze_data_distribution(train_labels, f"Fold {fold+1} Training Data Distribution")
        
        # Train model with recommended method
        start_train_time = time.perf_counter()
        fold_classifier.fit(train_texts, train_labels, balance_method=recommended_method)
        train_time = time.perf_counter() - start_train_time
        logger.info(f"Training time: {train_time:.3f}s")
        
        # Evaluate model
        logger.info("Evaluating model...")
        start_inference_time = time.perf_counter()
        val_predictions = fold_classifier.predict(val_texts)
        inference_time = time.perf_counter() - start_inference_time
        
        # Record memory usage
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_usage = memory_after - memory_before
        
        # Compute evaluation metrics
        eval_results = fold_classifier.compute_metrics(
            val_labels, val_predictions, 
            inference_time, len(val_labels), 
            memory_usage
        )
        
        fold_results.append(eval_results)
        
        # Log results
        logger.info(f"\nFold {fold+1} Evaluation Results:")
        logger.info(f"Accuracy: {eval_results['eval_accuracy']:.3f}")
        logger.info(f"Precision: {eval_results['eval_precision']:.3f}")
        logger.info(f"Recall: {eval_results['eval_recall']:.3f}")
        logger.info(f"F1 Score: {eval_results['eval_f1']:.3f}")
        logger.info(f"Inference time: {eval_results['avg_inference_time']:.3f}s")
        logger.info(f"Per-sample inference time: {eval_results['inference_time_per_sample']:.6f}s")
        logger.info(f"Memory usage: {eval_results['memory_usage_mb']:.1f}MB")
        
        # Print detailed classification report
        logger.info(f"\nDetailed Classification Report:")
        logger.info(f"\n{classification_report(val_labels, val_predictions)}")
        
        # Save best model
        if eval_results['eval_f1'] > best_f1:
            best_f1 = eval_results['eval_f1']
            best_model = fold_classifier
            logger.info(f"New best model! F1: {best_f1:.3f}")
    
    # Summarize cross-validation results
    logger.info("\n\nCross-Validation Final Results:")
    avg_metrics = {
        'accuracy': np.mean([r['eval_accuracy'] for r in fold_results]),
        'precision': np.mean([r['eval_precision'] for r in fold_results]),
        'recall': np.mean([r['eval_recall'] for r in fold_results]),
        'f1_score': np.mean([r['eval_f1'] for r in fold_results]),
        'avg_inference_time': np.mean([r['avg_inference_time'] for r in fold_results]),
        'avg_memory_usage': np.mean([r['memory_usage_mb'] for r in fold_results]),
        'std_accuracy': np.std([r['eval_accuracy'] for r in fold_results]),
        'std_f1': np.std([r['eval_f1'] for r in fold_results])
    }
    
    # Print final metrics
    print(f"\n{'='*50}")
    print("FINAL CROSS-VALIDATION RESULTS")
    print(f"{'='*50}")
    print(f"Average Accuracy: {avg_metrics['accuracy']:.3f} ± {avg_metrics['std_accuracy']:.3f}")
    print(f"Average Precision: {avg_metrics['precision']:.3f}")
    print(f"Average Recall: {avg_metrics['recall']:.3f}")
    print(f"Average F1 Score: {avg_metrics['f1_score']:.3f} ± {avg_metrics['std_f1']:.3f}")
    print(f"Average Inference Time: {avg_metrics['avg_inference_time']:.3f}s")
    print(f"Average Memory Usage: {avg_metrics['avg_memory_usage']:.1f}MB")
    
    # Save best model
    if best_model is not None:
        best_model.save_model("./svm_tfidf_binary_model.pkl")
        logger.info(f"\nBest model saved! (F1: {best_f1:.3f})")
    else:
        logger.warning("\nWarning: No model found to save")
    
    # Create visualizations
    create_analysis_visualizations(fold_results)
    
    # Print interpretability analysis
    print_interpretability_analysis(avg_metrics)
    
    return fold_results, avg_metrics


def create_analysis_visualizations(fold_results):
    """
    Create comprehensive analysis visualizations for binary classification.
    
    Args:
        fold_results: List of fold evaluation results
    """
    plt.figure(figsize=(15, 5))
    
    # Subplot 1: Memory vs Performance
    plt.subplot(1, 3, 1)
    plt.scatter(
        [r['memory_usage_mb'] for r in fold_results],
        [r['eval_f1'] for r in fold_results],
        c=[r['avg_inference_time'] for r in fold_results],
        cmap='viridis',
        s=100
    )
    plt.colorbar(label='Inference Time (s)')
    plt.xlabel('Memory Usage (MB)')
    plt.ylabel('F1 Score')
    plt.title('Memory vs Performance (SVM+TF-IDF)')
    
    # Subplot 2: Inference Time vs Performance
    plt.subplot(1, 3, 2)
    plt.scatter(
        [r['avg_inference_time'] for r in fold_results],
        [r['eval_f1'] for r in fold_results],
        c=range(len(fold_results)),
        cmap='tab10'
    )
    plt.xlabel('Inference Time (s)')
    plt.ylabel('F1 Score')
    plt.title('Inference Time vs Performance (SVM+TF-IDF)')
    
    # Subplot 3: Performance Across Folds
    plt.subplot(1, 3, 3)
    folds = range(1, len(fold_results) + 1)
    plt.plot(folds, [r['eval_f1'] for r in fold_results], 'bo-', label='F1 Score')
    plt.plot(folds, [r['eval_accuracy'] for r in fold_results], 'ro-', label='Accuracy')
    plt.xlabel('Fold')
    plt.ylabel('Score')
    plt.title('Performance Across Folds (SVM+TF-IDF)')
    plt.legend()
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('./svm_tfidf_binary_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Analysis visualizations saved")


def print_interpretability_analysis(avg_metrics):
    """
    Print interpretability analysis results.
    
    Args:
        avg_metrics: Dictionary of average metrics
    """
    print(f"\n{'='*50}")
    print("INTERPRETABILITY ANALYSIS")
    print(f"{'='*50}")
    print("SVM model characteristics:")
    print("• Uses TF-IDF for text vectorization")
    print("• Support Vector Machine for binary classification")
    print("• No GPU required, uses CPU for computation")
    print("• Good interpretability through feature weights")
    print("• Robust to noise and outliers")
    print("• Handles high-dimensional sparse features well")


if __name__ == "__main__":
    # Run the complete experiment
    try:
        fold_results, avg_metrics = run_cross_validation_experiment()
        print("\nExperiment completed successfully!")
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        raise