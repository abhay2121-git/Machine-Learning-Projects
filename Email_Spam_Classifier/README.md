# 📧 Email Spam Classifier

A production-grade machine learning project for detecting spam emails using Naive Bayes and Logistic Regression classifiers with interactive classification capabilities.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

## 🎯 Project Overview

This project implements a complete machine learning pipeline for email spam detection. It includes two classification models (Naive Bayes and Logistic Regression), comprehensive evaluation metrics, interactive email classification, and detailed visualizations for model analysis.

### **Key Features**
- ✅ **Dual Model Approach**: Naive Bayes & Logistic Regression
- ✅ **Interactive Classification**: Real-time spam detection for user inputs
- ✅ **Modular Architecture**: Clean, reusable code components
- ✅ **Production Ready**: Comprehensive testing and error handling
- ✅ **Rich Visualizations**: 5+ plots for data and model analysis
- ✅ **High Performance**: 96.52% accuracy with Logistic Regression

---

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Logistic Regression** | 96.52% | High | High | 0.96 |
| **Naive Bayes** | 88.12% | Moderate | High | 0.88 |

**Feature Count**: 1,764 features with word frequency analysis

---

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- 2+ GB disk space for dependencies and model files

---

## 🔧 Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/Email_Spam_Classifier.git
cd Email_Spam_Classifier
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Prepare Dataset
Add your email dataset at `data/dataset.csv` with the following format:
- **Column 1**: Email text content
- **Column 2**: Label (0 for ham, 1 for spam)

---

## 🚀 Quick Start

### Train Models
```bash
python main.py
# Select option 1 to train models
```

### Interactive Email Classification
```bash
python main.py
# Select option 2 to classify emails interactively
```

Or directly:
```bash
python text_classifier.py
```

### Run Demo
```bash
python demo_interactive.py
```

---

## 📁 Project Structure

```
Email_Spam_Classifier/
├── main.py                       # Main pipeline with menu system
├── text_classifier.py            # Interactive email classification module
├── predict.py                    # Prediction functionality
├── example_prediction.py          # Example prediction script
├── example_text_classification.py # Text classification example
├── demo_interactive.py           # Interactive demo
├── requirements.txt              # Project dependencies
├── README.md                     # This file
│
├── src/                          # Core ML modules
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── loader.py            # Data loading utilities
│   ├── features/
│   │   ├── __init__.py
│   │   └── vectorizer.py        # Text vectorization
│   ├── models/
│   │   ├── __init__.py
│   │   ├── naive_bayes.py       # Naive Bayes implementation
│   │   ├── logistic_regression.py # Logistic Regression implementation
│   │   └── trainer.py           # Model training utilities
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluator.py         # Evaluation metrics
│   └── visualization/
│       ├── __init__.py
│       └── plotter.py           # Visualization utilities
│
├── data/                         # Dataset directory
│   └── dataset.csv              # Email dataset (add yours here)
│
├── models/                       # Trained models (auto-generated)
│   ├── logistic_regression.pkl
│   ├── naive_bayes.pkl
│   └── vectorizer.pkl
│
├── outputs/                      # Model outputs (auto-generated)
│   ├── models/
│   └── plots/
│
└── tests/                        # Unit tests
    ├── test_loader.py
    ├── test_preprocessor.py
    ├── test_vectorizer.py
    ├── test_naive_bayes.py
    └── test_logistic_regression.py
```

---

## 📖 Usage

### 1. Train Models
```bash
python main.py
# Follow the menu: Select 1 to train models
```

### 2. Classify Emails Interactively
```bash
python main.py
# Follow the menu: Select 2 to classify emails
# Enter your email text and receive spam/ham prediction
```

### 3. Run Example Predictions
```bash
python example_prediction.py
python example_text_classification.py
```

### 4. Run Tests
```bash
pytest tests/
# or
python -m pytest tests/ -v
```

---

## 📚 Dataset Information

- **Source**: Email dataset with spam/ham classification
- **Format**: CSV with email text and labels
- **Size**: Adjustable based on your dataset
- **Labels**: 
  - `0` = Ham (legitimate email)
  - `1` = Spam

Place your dataset at: `data/dataset.csv`

---

## 🔬 Technical Details

### Data Preprocessing
- Text tokenization and cleaning
- Stop word removal
- Stemming/Lemmatization
- TF-IDF vectorization (1,764 features)

### Models Used
1. **Naive Bayes**: Fast, probabilistic classifier
2. **Logistic Regression**: High-performance linear classifier

### Evaluation Metrics
- Accuracy
- Precision & Recall
- F1-Score
- Confusion Matrix
- ROC-AUC Curve

---

## 📈 Visualizations

The project generates 5+ visualization plots:
- Confusion Matrices
- ROC Curves
- Feature Importance
- Classification Reports
- Distribution Analysis

View outputs in: `outputs/plots/`

---

## ⚙️ Configuration

Key parameters can be modified in source files:
- **Training/Test Split**: `src/data/loader.py`
- **Vectorizer Settings**: `src/features/vectorizer.py`
- **Model Parameters**: `src/models/naive_bayes.py`, `src/models/logistic_regression.py`

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Steps to Contribute:
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👨‍💻 Author

**Abhay** - [GitHub Profile](https://github.com/abhay2121-git)

---

## 📞 Support & Issues

If you encounter any issues or have questions:
- Open an [Issue](https://github.com/YOUR_USERNAME/Email_Spam_Classifier/issues)
- Check existing documentation in `src/` module files
- Review test files for usage examples

---

## 🎓 Learning Resources

This project demonstrates:
- Machine Learning classification pipelines
- Text preprocessing and feature engineering
- Model training, evaluation, and comparison
- Interactive Python applications
- Unit testing best practices
- Modular code architecture

---

**Last Updated**: May 2026 | **Status**: Production Ready ✅
