# Multimodal Authentication System Using Face and Voice Data
---

# Project Overview

This project presents a multimodal biometric authentication system that integrates face and voice data through feature-level fusion. It improves upon unimodal systems by leveraging the complementary strengths of both modalities. The system uses machine learning classifiers—SVM, Random Forest, and kNN—and addresses class imbalance using SMOTE.

---

# Research Questions

1. How does feature-level fusion affect authentication performance?
2. What is the impact of SMOTE on class balancing?
3. Which classifier performs best with multimodal features?

---

# Project Structure

.
├── Biometrics.py                          # Main script with data processing, feature extraction, training & evaluation
├── Multimodal_Auth.ipynb                  # Notebook version for experimentation and visualization
├── Multimodal_Authentication_System_Using_Face_and_Voice_Data.pdf  # Detailed report
├── Multimodal_UserAuthentication.pptx     # Final presentation
├── README.md                              # Project documentation


 ---

# Datasets

- Face: [Caltech Face Dataset](http://www.vision.caltech.edu/Image_Datasets/faces/)
- Voice: [AudioMNIST Dataset](https://github.com/soerenab/AudioMNIST)

Update dataset paths in `Biometrics.py`:

FACE_DATA_DIR = "<path_to_face_dataset>"
VOICE_DATA_DIR = "<path_to_voice_dataset>"


---

# Requirements

Install dependencies with:


pip install numpy pandas opencv-python librosa scikit-learn imbalanced-learn matplotlib seaborn

---

# How to Run

# Option 1: Python Script

python Biometrics.py

# Option 2: Jupyter Notebook

Open and run `Multimodal_Auth.ipynb`.

# Results Summary

| System     | Accuracy | ROC AUC | EER     | D-prime |
|------------|----------|---------|---------|---------|
| Face-Only  | 0.99     | 1.00    | 0.0001  | 11.98   |
| Voice-Only | 0.95     | 0.99    | 0.0149  | 4.93    |
| Multimodal | 0.99     | 1.00    | 0.0001  | 12.22   |

- SVM classifier achieved the best overall performance.

---

# Techniques Used

- Feature Extraction:
  - Face: Pixel intensity + Canny edge detection
  - Voice: MFCC + Spectral Contrast
- Fusion: Feature-level concatenation
- Balancing: SMOTE
- Evaluation: Accuracy, ROC AUC, EER, D-prime, Confusion Matrix

---

# Ethical Considerations

- Encrypted biometric data handling
- Fairness audits for demographic bias
- Transparency and user consent emphasized

---

# Future Improvements

- Incorporate score-level and decision-level fusion
- Test under real-world noisy and lighting conditions
- Include more biometric modalities

---
