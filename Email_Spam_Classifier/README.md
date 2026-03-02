# Email Spam Classifier v1.0

A complete, production-grade machine learning pipeline for email spam detection using Naive Bayes and Logistic Regression classifiers.

## 🚀 Features

- **Dual Model Approach**: Implements both Multinomial Naive Bayes and Logistic Regression
- **Comprehensive Evaluation**: Detailed metrics comparison and performance analysis
- **Rich Visualizations**: 5 different plots for model analysis and interpretation
- **Production Ready**: Modular, well-documented, and easily extensible codebase
- **Data Pipeline**: Complete end-to-end processing from raw data to trained models

## 📋 Requirements

- Python 3.11+
- See `requirements.txt` for package dependencies

## 🛠️ Installation

1. Clone or download this repository
2. Navigate to the project directory
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 📁 Project Structure

```
Email_Spam_Classifier/
├── data/
│   └── dataset.csv                    # Your dataset goes here
│
├── notebooks/
│   └── exploration.ipynb              # EDA and experimentation
│
├── src/
│   ├── data/
│   │   ├── loader.py                  # Data loading
│   │   └── preprocessor.py           # Data preprocessing
│   │
│   ├── features/
│   │   └── vectorizer.py             # TF-IDF vectorization
│   │
│   ├── models/
│   │   ├── naive_bayes.py            # Naive Bayes implementation
│   │   ├── logistic_regression.py    # Logistic Regression implementation
│   │   └── trainer.py               # Unified training pipeline
│   │
│   ├── evaluation/
│   │   └── evaluator.py             # Model evaluation and comparison
│   │
│   └── visualization/
│       └── plotter.py               # All visualization plots
│
├── outputs/
│   ├── models/                       # Trained models (.pkl files)
│   └── plots/                        # Generated plots (.png files)
│
├── tests/                            # Unit tests
├── main.py                           # Main pipeline entry point
├── requirements.txt                  # Dependencies
└── README.md                         # This file
```

## 📊 Dataset Requirements

Your dataset should be a CSV file with at least 2 columns:

1. **Text Column**: Contains the email content/message
2. **Label Column**: Contains the classification (spam/ham or 1/0)

**Example format:**
```
message,label
"Free Viagra now!",spam
"Hello friend, how are you?",ham
"Claim your prize!",spam
"Meeting tomorrow at 3pm",ham
```

**Place your dataset as `data/dataset.csv`**

## 🚀 Quick Start

1. **Add your dataset** to `data/dataset.csv`
2. **Run the main program**:
   ```bash
   python main.py
   ```

**Main Program Options:**
- **Option 1**: Train Models (Full Pipeline) - Trains and evaluates both models
- **Option 2**: Classify Email (Interactive Mode) - Classify emails in real-time
- **Option 3**: Exit

The pipeline will automatically:
- Load and preprocess your data
- Train both Naive Bayes and Logistic Regression models
- Evaluate and compare their performance
- Generate comprehensive visualizations
- Save trained models and plots to the `outputs/` directory

## 📧 Making Predictions

### **🔥 NEW: Interactive Mode in main.py (Recommended)**
```bash
python main.py
```
Choose **Option 2** for interactive email classification:
- **Paste actual email text** directly into the terminal
- **Automatic text processing** (cleaning, stopword removal, word extraction)
- **Real-time classification** with confidence scores
- **Multi-line email input** support

**Example Usage:**
```
Choose an option:
1. Train Models (Full Pipeline)
2. Classify Email (Interactive Mode)
3. Exit

Enter your choice (1, 2, or 3): 2

Enter your email text:
CONGRATULATIONS! You have won a FREE vacation!
Click here to claim your prize immediately.
Limited time offer - don't miss out!

Classification: Spam (Confidence: 97.76%)
```

### **Interactive Prediction (Word Frequencies)**
```bash
python predict.py
```
- Enter word frequencies in format: `word:count word:count`
- Choose between Logistic Regression (96.52% accuracy) and Naive Bayes (88.12% accuracy)
- Get instant classification with confidence scores

### **Programmatic Prediction**
```python
from text_classifier import classify_email_text

# Classify email text directly
email_text = "Congratulations! You won a free vacation!"
result, confidence = classify_email_text(email_text, "logistic")
print(f"Classification: {result}")
print(f"Confidence: {confidence:.2f}%")
```

### **Example Usage**
```bash
# Test with example emails
python example_text_classification.py

# Test with word frequencies (old method)
python example_prediction.py
```

**Sample Output:**
```
Testing SPAM email:
Text: CONGRATULATIONS! You have won a FREE vacation...
Prediction: Spam (Confidence: 97.76%)

Testing HAM email:
Text: Hi team, Just wanted to remind everyone about...
Prediction: Ham (Confidence: 56.87%)
```

## �� Pipeline Steps

### 1. Data Loading (`src/data/loader.py`)
- Loads CSV dataset with error handling
- Displays dataset information and statistics

### 2. Data Preprocessing (`src/data/preprocessor.py`)
- Removes duplicates and null values
- Cleans text (removes special characters, converts to lowercase)
- Removes stopwords
- Converts labels to binary (spam=1, ham=0)
- Splits data into train/test sets (80/20, stratified)

### 3. Feature Engineering (`src/features/vectorizer.py`)
- Applies TF-IDF vectorization (max 5000 features)
- Uses n-grams (1-2) for better context
- Saves fitted vectorizer for future use

### 4. Model Training (`src/models/`)
- **Naive Bayes**: MultinomialNB with alpha=1.0
- **Logistic Regression**: L2 regularization, max_iter=1000
- Both models saved as `.pkl` files

### 5. Model Evaluation (`src/evaluation/evaluator.py`)
- Comprehensive metrics: Accuracy, Precision, Recall, F1, ROC AUC
- Detailed classification reports
- Confusion matrices
- Performance comparison table
- Winner analysis

### 6. Visualization (`src/visualization/plotter.py`)
Generates 5 plots:
- **Class Distribution**: Spam vs Ham count
- **Confusion Matrices**: For both models
- **ROC Curve**: Comparison of both models
- **Feature Importance**: Top 20 predictive words

## 📊 Output Files

### Trained Models (`outputs/models/`)
- `naive_bayes_model.pkl` - Trained Naive Bayes classifier
- `logistic_regression_model.pkl` - Trained Logistic Regression classifier
- `vectorizer.pkl` - Fitted TF-IDF vectorizer

### Visualizations (`outputs/plots/`)
- `class_distribution.png` - Dataset class balance
- `confusion_matrix_nb.png` - Naive Bayes confusion matrix
- `confusion_matrix_lr.png` - Logistic Regression confusion matrix
- `roc_curve.png` - ROC curve comparison
- `feature_importance.png` - Top 20 predictive features

## 🧪 Testing

Run the test suite:
```bash
python -m pytest tests/
```

## 📓 Jupyter Notebook

For exploratory data analysis and experimentation:
```bash
jupyter notebook notebooks/exploration.ipynb
```

## 🔧 Customization

### Adding New Models
1. Create new model file in `src/models/`
2. Implement training and prediction functions
3. Update `src/models/trainer.py` to include new model
4. Update evaluation and visualization modules

### Changing Vectorization Parameters
Edit `src/features/vectorizer.py`:
- `max_features`: Number of features (default: 5000)
- `ngram_range`: N-gram range (default: (1, 2))
- `min_df`/`max_df`: Document frequency thresholds

### Modifying Preprocessing
Edit `src/data/preprocessor.py` to:
- Add custom text cleaning steps
- Modify stopwords list
- Change train-test split ratio

## 🐛 Troubleshooting

### Common Issues

1. **Dataset not found**
   - Ensure `data/dataset.csv` exists
   - Check file path and permissions

2. **Column not found**
   - Verify your dataset has text and label columns
   - Check column names match expected format

3. **Memory issues**
   - Reduce `max_features` in vectorizer
   - Use smaller dataset sample

4. **Import errors**
   - Run `pip install -r requirements.txt`
   - Check Python version (3.11+)

### Performance Tips

- For large datasets, consider reducing `max_features`
- Use `min_df` parameter to filter rare terms
- Experiment with different `ngram_range` values
- Try different regularization parameters for Logistic Regression

## 📚 Algorithm Details

### Naive Bayes (Multinomial)
- Based on Bayes' theorem with independence assumptions
- Works well with text classification
- Uses term frequency features
- Fast training and prediction

### Logistic Regression
- Linear classification model
- Uses L2 regularization to prevent overfitting
- Provides probability estimates
- Interpretable coefficients

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Scikit-learn for machine learning algorithms
- Matplotlib and Seaborn for visualization
- Pandas for data manipulation
- NumPy for numerical operations

---

**Built with ❤️ for Machine Learning Engineers**
