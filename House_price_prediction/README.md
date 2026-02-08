# House Price Prediction

A machine learning project for predicting house prices using various regression models.

## Project Structure

```
house_price_prediction/
│
├── src/                    # Core ML modules
│   ├── data_loader.py      # Data loading and inspection
│   ├── preprocessing.py    # Data cleaning and preprocessing
│   ├── model.py           # ML model definitions
│   ├── train.py           # Training pipeline
│   └── evaluate.py        # Model evaluation utilities
│
├── visual/                 # Visualization modules
│   ├── eda_plots.py       # Exploratory data analysis plots
│   └── model_plots.py     # Model evaluation plots
│
├── data/                   # Dataset files
│   └── (add your dataset here)
│
├── artifacts/              # Trained models and scalers (auto-created)
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Installation

1. Clone or download this project
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Prepare your data
Place your dataset CSV file in the `data/` directory. The dataset should have:
- Features columns (numerical and categorical)
- A target column (house price)

### 2. Train the model
```python
from src.train import train_model

# Train a random forest model
results = train_model(
    data_path='data/your_dataset.csv',
    target_column='price',  # Change to your target column name
    model_type='rf'         # Options: 'linear', 'ridge', 'lasso', 'rf', 'gbm'
)

print(f"Test R²: {results['test_metrics']['r2']:.4f}")
print(f"Test RMSE: {results['test_metrics']['rmse']:.4f}")
```

### 3. Evaluate the model
```python
from src.evaluate import load_model, load_scaler, evaluate_model
import pandas as pd

# Load trained artifacts
model = load_model()
scaler = load_scaler()

# Load test data
test_data = pd.read_csv('data/test.csv')
X_test = test_data.drop(columns=['price'])
y_test = test_data['price']

# Evaluate
metrics = evaluate_model(model, X_test, y_test)
print(metrics)
```

### 4. Visualize results
```python
from visual.eda_plots import generate_eda_report
from visual.model_plots import plot_predictions_vs_actual

# EDA plots
df = pd.read_csv('data/your_dataset.csv')
generate_eda_report(df, 'price')

# Model evaluation plots
y_pred = model.predict(X_test)
plot_predictions_vs_actual(y_test.values, y_pred)
```

## Available Models

- **Linear Regression**: Simple baseline model
- **Ridge Regression**: Linear regression with L2 regularization
- **Lasso Regression**: Linear regression with L1 regularization
- **Random Forest**: Ensemble decision tree model
- **Gradient Boosting**: Boosted ensemble model

## Features

- **Modular Design**: Separate modules for data loading, preprocessing, modeling, and evaluation
- **Multiple Models**: Support for various regression algorithms
- **Comprehensive Evaluation**: MSE, RMSE, MAE, R² metrics
- **Visualization**: EDA plots and model evaluation visualizations
- **Artifact Management**: Automatic saving of trained models and scalers

## File Descriptions

### Core Modules (`src/`)

- **`data_loader.py`**: Handles CSV loading and basic data inspection
- **`preprocessing.py`**: Missing value handling, categorical encoding, feature scaling
- **`model.py`**: ML model wrapper with training, prediction, and evaluation methods
- **`train.py`**: Complete training pipeline orchestrating all steps
- **`evaluate.py`**: Model evaluation, residual analysis, and prediction utilities

### Visualization (`visual/`)

- **`eda_plots.py`**: Distribution plots, correlation matrices, missing value analysis
- **`model_plots.py`**: Prediction vs actual plots, residual analysis, feature importance

## Getting Started

1. **Start with EDA**: Use `visual/eda_plots.py` to understand your data
2. **Preprocess data**: The `src/preprocessing.py` module handles cleaning
3. **Train models**: Use `src/train.py` to train different models
4. **Evaluate performance**: Use `src/evaluate.py` and `visual/model_plots.py` for analysis

## Best Practices Followed

- ✅ **Separation of Concerns**: Clear separation between data, model, and visualization logic
- ✅ **Modular Design**: Each file has a single responsibility
- ✅ **Type Hints**: All functions include type annotations
- ✅ **Documentation**: Comprehensive docstrings for all functions
- ✅ **Artifact Management**: Models and preprocessors are saved for reproducibility
- ✅ **Error Handling**: Proper validation and error messages
- ✅ **Scalability**: Easy to add new models and preprocessing steps

## Next Steps

1. Add your dataset to the `data/` directory
2. Start with exploratory data analysis using `visual/eda_plots.py`
3. Train your first model using `src/train.py`
4. Evaluate and visualize results using `src/evaluate.py` and `visual/model_plots.py`
