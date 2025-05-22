#!/usr/bin/env python
# coding: utf-8
########################################################################
#Research questions
'''RQ1: How does feature-level fusion of facial and voice
biometric data impact the overall system performance
compared to unimodal systems?

Feature-level fusion integrates complementary informa-
tion from face and voice modalities, leading to improved
authentication accuracy. Experiments show that multi-
modal systems outperform unimodal systems in Equal
Error Rate (EER) and ROC AUC metrics, demonstrating
the robustness of the fused approach.

RQ2: How does the use of SMOTE for balancing class
distributions affect the performance of classification
models in multimodal biometric authentication?

SMOTE effectively mitigates class imbalance, resulting
in improved precision, recall, and F1-scores across all
classes. Models trained on balanced datasets exhibited
consistent performance enhancements, particularly in
reducing false negatives for underrepresented classes.

RQ3: Which classifier (Random Forest, SVM, or kNN)
performs best in a multimodal biometric authentication
system based on facial and voice features?

SVM demonstrated the highest performance in the
multimodal biometric authentication system, achieving
the best accuracy and separability due to its efficiency
with high-dimensional data. Random Forest ranked
second, leveraging complex feature interactions, while
kNN showed limitations in scalability and parameter
sensitivity.'''

# Loading Librarires 
import os
import numpy as np
import pandas as pd
import cv2
import librosa
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, auc
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, label_binarize
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import random
import warnings

warnings.filterwarnings("ignore")  # Suppress warnings globally


# Data Paths

# Paths to the face and voice datasets
FACE_DATA_DIR = "C:/Users/sruth/Downloads/lokesg/face"
VOICE_DATA_DIR = "C:/Users/sruth/Downloads/lokesg/Audio"


# Confusion Matrix 

# Plots the confusion matrix for visual evaluation of model predictions using a heatmap.
# Parameters:
# - cm: Confusion matrix
# - classes: Class labels

# Helper Functions
def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()


def plot_multiclass_roc_curve(classifier, X_test, y_test, classes):
    # Multiclass ROC-AUC curve
    # Plots the ROC curve for multiclass classification.
    # Parameters:
    # - classifier: Trained classifier
    # - X_test: Test features
    # - y_test: True labels
    # - classes: Class labels
    y_test_binarized = label_binarize(y_test, classes=classes)
    y_score = classifier.predict_proba(X_test)
 
    fpr, tpr, roc_auc = {}, {}, {}
    for i, class_label in enumerate(classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
 
    plt.figure()
    for i, class_label in enumerate(classes):
        plt.plot(fpr[i], tpr[i], label=f"Class {class_label} (AUC = {roc_auc[i]:.2f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve for Multiclass Classification")
    plt.legend(loc="lower right")
    plt.show()
 
    macro_auc = roc_auc_score(y_test_binarized, y_score, average="macro")
    print(f"Macro-average ROC AUC: {macro_auc:.2f}")


# EER and d-prime 

# calculate_eer
# Calculates the Equal Error Rate (EER) from FPR and TPR values.
# Parameters:
# - fpr: False Positive Rates
# - tpr: True Positive Rates
# - thresholds: Threshold values
# Returns:
# - eer: Equal Error Rate
# - eer_threshold: Threshold corresponding to EER


def calculate_eer(fpr, tpr, thresholds):
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.abs(fpr - fnr))]
    eer = fpr[np.nanargmin(np.abs(fpr - fnr))]
    return eer, eer_threshold

# calculate_d_prime
# 
# Calculates the D-prime value to measure separation between genuine and impostor distributions.
# Parameters:
# - genuine_scores: Scores for genuine samples
# - impostor_scores: Scores for impostor samples
# Returns:
# - d_prime: D-prime value
 
def calculate_d_prime(genuine_scores, impostor_scores):
    mean_genuine = np.mean(genuine_scores)
    mean_impostor = np.mean(impostor_scores)
    std_genuine = np.std(genuine_scores)
    std_impostor = np.std(impostor_scores)
    d_prime = (mean_genuine - mean_impostor) / np.sqrt(0.5 * (std_genuine**2 + std_impostor**2))
    return d_prime


# plot_score_distribution
# 
# Plots the score distribution for genuine and impostor samples.
# Parameters:
# - genuine_scores: Scores for genuine samples
# - impostor_scores: Scores for impostor samples
 
def plot_score_distribution(genuine_scores, impostor_scores):
    plt.figure(figsize=(10, 6))
    sns.histplot(genuine_scores, color='green', label='Genuine', kde=True, stat='density')
    sns.histplot(impostor_scores, color='red', label='Impostor', kde=True, stat='density')
    plt.xlabel("Score")
    plt.ylabel("Density")
    plt.title("Score Distribution")
    plt.legend()
    plt.show()


# Data Augumentation

# Feature Extraction Functions:
# 
# The extract_edge_features function processes an image to detect edges using the Canny edge detection method. This is useful for capturing structural details of the image, which can serve as robust features for image classification or recognition tasks.
# The extract_spectral_contrast function extracts audio features related to the contrast between spectral peaks and valleys, helping analyze and classify audio signals.
# 
# Image Augmentation:
# 
# The augment_image function applies a series of transformations to an image, including rotation, flipping, and brightness adjustments. These augmentations introduce variability to the dataset, making the model more robust to different orientations and lighting conditions.
# 
# Face Data Augmentation Pipeline:
# 
# The augment_face_data function creates a more diverse training dataset by augmenting each input face image multiple times. It combines the augmented image data with edge features to create enriched feature vectors. This approach improves the quality and quantity of the dataset, helping to prevent overfitting in machine learning models.


def extract_edge_features(image):
    image_8bit = (image * 255).astype(np.uint8)
    edges = cv2.Canny(image_8bit, threshold1=100, threshold2=200)
    return edges.flatten()
 
def extract_spectral_contrast(signal, sr):
    return librosa.feature.spectral_contrast(y=signal, sr=sr).flatten()
 
def augment_image(image):
    rows, cols = image.shape
    angle = random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated = cv2.warpAffine(image, M, (cols, rows))
    flip = random.choice([-1, 0, 1])  # Horizontal, vertical, or both
    flipped = cv2.flip(rotated, flip)
    brightness = random.uniform(0.8, 1.2)
    adjusted = np.clip(flipped * brightness, 0, 1)
    return adjusted
 
def augment_face_data(X_faces, y_faces):
    augmented_faces, augmented_labels = [], []
    for image, label in zip(X_faces, y_faces):
        image = image.reshape(128, 128)
        edge_features = extract_edge_features(image)
        for _ in range(3):
            augmented_faces.append(np.hstack((augment_image(image).flatten(), edge_features)))
            augmented_labels.append(label)
    return np.array(augmented_faces), np.array(augmented_labels)


# Loading the Data

# create_face_labels:
# Loads face images, resizes, and normalizes them.
# Extracts and processes face images from the dataset.
# Returns:
# - Processed face images and their corresponding labels
# 

def create_face_labels(directory):
    images, labels = [], []
    for label in os.listdir(directory):
        label_path = os.path.join(directory, label)
        if os.path.isdir(label_path):
            for file in os.listdir(label_path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img = cv2.imread(os.path.join(label_path, file), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img_resized = cv2.resize(img, (128, 128)) / 255.0
                        images.append(img_resized.flatten())
                        labels.append(label)
    print(f"Loaded {len(images)} face images with {len(set(labels))} unique labels.")
    return np.array(images), np.array(labels)

# process_voice:
# 
# Processes audio files into MFCC and spectral contrast features, normalizing lengths.
# Extracts and processes voice features from the dataset.
# Returns:
# - Processed voice features and their corresponding labels
 
def process_voice(directory):
    voice_features, voice_labels = [], []
    max_length = 300
    for label in os.listdir(directory):
        label_path = os.path.join(directory, label)
        if os.path.isdir(label_path):
            for file in os.listdir(label_path):
                if file.lower().endswith('.wav'):
                    path = os.path.join(label_path, file)
                    try:
                        signal, sr = librosa.load(path, sr=16000)
                        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13).flatten()
                        spectral_contrast = extract_spectral_contrast(signal, sr)
                        combined_features = np.hstack((mfcc, spectral_contrast))
                        if len(combined_features) < max_length:
                            combined_features = np.pad(combined_features, (0, max_length - len(combined_features)), mode="constant")
                        elif len(combined_features) > max_length:
                            combined_features = combined_features[:max_length]
                        voice_features.append(combined_features)
                        voice_labels.append(label)
                    except Exception as e:
                        print(f"Error processing file {path}: {e}")
    print(f"Voice features extracted: {len(voice_features)}")
    return np.array(voice_features), np.array(voice_labels)


# Aligning Face data and Voice data for common users
#This function identifies users common to both face and voice datasets, aligns their 
#data, and returns subsets of features and labels for those users. This ensures that 
#multimodal biometric analysis is performed only on users available in both modalities.

def align_face_voice_data(X_faces, y_faces, X_voices, y_voices):
    common_labels = set(y_faces) & set(y_voices)
    print(f"Common users: {len(common_labels)}")
   
    X_faces_aligned, y_faces_aligned = [], []
    X_voices_aligned, y_voices_aligned = [], []
   
    for label in common_labels:
        face_indices = [i for i, y in enumerate(y_faces) if y == label]
        voice_indices = [i for i, y in enumerate(y_voices) if y == label]
        min_samples = min(len(face_indices), len(voice_indices))
        X_faces_aligned.extend(X_faces[face_indices[:min_samples]])
        y_faces_aligned.extend([label] * min_samples)
        X_voices_aligned.extend(X_voices[voice_indices[:min_samples]])
        y_voices_aligned.extend([label] * min_samples)
   
    print(f"Aligned Face Data Shape: {np.array(X_faces_aligned).shape}")
    print(f"Aligned Voice Data Shape: {np.array(X_voices_aligned).shape}")
    return (
        np.array(X_faces_aligned),
        np.array(y_faces_aligned),
        np.array(X_voices_aligned),
        np.array(y_voices_aligned),
    )


# Visulaize the Results
#Visualizes performance metrics across cross-validation folds.
#
#This function plots key metrics such as accuracy, ROC AUC, EER, and d-prime
#for a given classifier across multiple folds during cross-validation. It helps 
#in understanding model consistency.
#
#Parameters:
#- fold_metrics: dict, Dictionary containing metric values for each fold
#- classifier_name: str, Name of the classifier being evaluated

def visualize_results(fold_metrics, classifier_name):
    """Visualize fold metrics for a given classifier."""
    metrics = ['accuracy', 'macro_roc_auc', 'average_eer', 'd_prime']
    titles = ['Accuracy', 'Macro-average ROC AUC', 'Average EER', 'D-prime']
    y_labels = ['Accuracy', 'ROC AUC', 'EER', 'D-prime']
    
    for metric, title, y_label in zip(metrics, titles, y_labels):
        plt.figure(figsize=(8, 6))
        plt.plot(fold_metrics[metric], marker='o', linestyle='-', label=f'{metric} per fold')
        plt.axhline(y=np.mean(fold_metrics[metric]), color='r', linestyle='--', label=f'Mean {title}')
        plt.title(f'{title} across Folds for {classifier_name}')
        plt.xlabel('Fold')
        plt.ylabel(y_label)
        plt.legend()
        plt.grid(True)
        plt.show()


# The function compare_and_visualize provides an effective mechanism to compare and visualize performance metrics across different systems (Face-Only, Voice-Only, and Multimodal). It facilitates both numerical and visual analysis, which is essential for performance evaluation in machine learning projects.
#Parameters:
#    - unimodal_metrics: dict, Performance metrics for unimodal systems (Face and Voice)
#    - multimodal_metrics: list, Performance metrics for the multimodal system
#    - metric_names: list, Names of the metrics being compared

def compare_and_visualize_results(unimodal_metrics, multimodal_metrics, metric_names):
    """Compare and visualize unimodal vs. multimodal metrics."""
    comparison_df = pd.DataFrame({
        "Metric": metric_names,
        "Face-Only": unimodal_metrics["Face"],
        "Voice-Only": unimodal_metrics["Voice"],
        "Multimodal": multimodal_metrics
    })
    
    # Print comparison table
    print("\nPerformance Comparison (Unimodal vs. Multimodal):")
    print(comparison_df.to_string(index=False))

    # Bar plot for visualization
    comparison_df.plot(
        x="Metric",
        kind="bar",
        figsize=(12, 6),
        title="Performance Comparison (Unimodal vs. Multimodal)"
    )
    plt.ylabel("Score")
    plt.xticks(rotation=0)
    plt.legend(title="System")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()


# Training the model and evaluates models using stratified K-Fold cross-validation
#Parameters:
#    - X_combined: np.array, Combined features (Face + Voice)
#    - y_combined: np.array, Labels for the combined features
#    - n_splits: int, Number of folds for cross-validation (default=5)

#    Returns:
#    - overall_metrics: dict, Averaged performance metrics for all folds
from sklearn.model_selection import StratifiedKFold

def train_and_evaluate_with_cv(X_combined, y_combined, n_splits=5):
    # Ensure balanced classes using SMOTE
    class_counts = Counter(y_combined)
    min_samples = min(class_counts.values())
    smote_neighbors = min(5, min_samples - 1) if min_samples > 1 else 1
    smote = SMOTE(random_state=42, k_neighbors=smote_neighbors)
    try:
        X_balanced, y_balanced = smote.fit_resample(X_combined, y_combined)
    except ValueError as e:
        print(f"Error during SMOTE resampling: {e}")
        return

    # Initialize Stratified K-Fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Initialize classifiers
    classifiers = {
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "SVM": SVC(kernel='linear', probability=True, random_state=42),
        "kNN": KNeighborsClassifier(n_neighbors=5),
    }
    overall_metrics = {}
    for name, clf in classifiers.items():
        print(f"\nEvaluating {name} with {n_splits}-fold cross-validation...")
        fold_metrics = {"accuracy": [], "macro_roc_auc": [], "average_eer": [], "d_prime": []}

        for train_idx, test_idx in skf.split(X_balanced, y_balanced):
            X_train, X_test = X_balanced[train_idx], X_balanced[test_idx]
            y_train, y_test = y_balanced[train_idx], y_balanced[test_idx]

            # Feature scaling
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Train the model
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            # Classification report
            print(f"\nClassification Report for Fold:")
            print(classification_report(y_test, y_pred))

            # Confusion matrix visualization
            cm = confusion_matrix(y_test, y_pred)
            plot_confusion_matrix(cm, classes=np.unique(y_faces_aligned))

            # Multiclass ROC Curve
            if hasattr(clf, "predict_proba"):
                y_score = clf.predict_proba(X_test)
                plot_multiclass_roc_curve(clf, X_test, y_test, classes=np.unique(y_faces_aligned))

                # Compute ROC and EER
                y_test_binarized = label_binarize(y_test, classes=np.unique(y_combined))
                n_classes = y_test_binarized.shape[1]
                fpr, tpr, thresholds = {}, {}, {}
                eer_values = []
                genuine_scores, impostor_scores = [], []

                for i in range(n_classes):
                    fpr[i], tpr[i], thresholds[i] = roc_curve(y_test_binarized[:, i], y_score[:, i])
                    eer, eer_threshold = calculate_eer(fpr[i], tpr[i], thresholds[i])
                    eer_values.append(eer)

                    genuine_scores.extend(y_score[y_test_binarized[:, i] == 1, i])
                    impostor_scores.extend(y_score[y_test_binarized[:, i] == 0, i])


                
                # Step 12: Score distribution visualization
                plot_score_distribution(genuine_scores, impostor_scores)

                avg_eer = np.mean(eer_values)
                macro_roc_auc = roc_auc_score(y_test_binarized, y_score, average="macro")
                d_prime_value = calculate_d_prime(np.array(genuine_scores), np.array(impostor_scores))

                fold_metrics["accuracy"].append(np.mean(y_test == y_pred))
                fold_metrics["macro_roc_auc"].append(macro_roc_auc)
                fold_metrics["average_eer"].append(avg_eer)
                fold_metrics["d_prime"].append(d_prime_value)

        #Report fold averages
        print(f"\n{name} Cross-Validation Results:")
        print(f"Mean Accuracy: {np.mean(fold_metrics['accuracy']):.4f}")
        print(f"Mean Macro ROC AUC: {np.mean(fold_metrics['macro_roc_auc']):.4f}")
        print(f"Mean Average EER: {np.mean(fold_metrics['average_eer']):.4f}")
        print(f"Mean D-prime: {np.mean(fold_metrics['d_prime']):.4f}")
        overall_metrics = {key: np.mean(value) for key, value in fold_metrics.items()}
        #visualize results for the classifier
        visualize_results(fold_metrics, name)
        
    return overall_metrics


# Main Execution
X_faces, y_faces = create_face_labels(FACE_DATA_DIR)
X_faces_aug, y_faces_aug = augment_face_data(X_faces, y_faces)
X_voices, y_voices = process_voice(VOICE_DATA_DIR)
X_faces_aligned, y_faces_aligned, X_voices_aligned, y_voices_aligned = align_face_voice_data(
    X_faces_aug, y_faces_aug, X_voices, y_voices
)
X_combined = np.hstack((X_faces_aligned, X_voices_aligned))


# Evaluating Only Face System

print("\nEvaluating Face-Only System...")
face_metrics = train_and_evaluate_with_cv(X_faces_aligned, y_faces_aligned, n_splits=5)


# Evaluating Only Voice system

print("\nEvaluating Voice-Only System...")
voice_metrics = train_and_evaluate_with_cv(X_voices_aligned, y_voices_aligned, n_splits=5)


# Multimodal Evaluation - Evaluating with both the Face data and Voice data

print("\nEvaluating Multimodal System...")
multimodal_metrics = train_and_evaluate_with_cv(X_combined, y_faces_aligned, n_splits=5) 


unimodal_metrics = {
    "Face": list(face_metrics.values()),
    "Voice": list(voice_metrics.values())
}
multimodal_values = list(multimodal_metrics.values())
metric_names = list(face_metrics.keys())

# Compare and visualize results
compare_and_visualize_results(unimodal_metrics, multimodal_values, metric_names)
