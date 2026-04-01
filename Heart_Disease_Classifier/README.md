# ❤️ Heart Disease Classifier

A machine learning project to predict heart disease risk using clinical data. Implements multiple classification algorithms and provides a Streamlit web application for predictions.

## 📋 Quick Start

### Installation
```bash
cd Heart_Disease_Classifier
pip install -r requirements.txt
```

### Usage

**Run the ML pipeline:**
```bash
python main.py
```

**Launch the web app:**
```bash
streamlit run app.py
```

## 📊 Dataset

- **Source**: UCI Heart Disease Dataset (~303 samples)
- **Features**: 13 clinical attributes (age, sex, chest pain, BP, cholesterol, etc.)
- **Target**: Heart Disease (0=No, 1=Yes)

## 🤖 Models

The project trains and compares 5 algorithms:
- Logistic Regression
- Naive Bayes
- Support Vector Machine
- Decision Tree
- XGBoost

## 📁 Project Structure

```
Heart_Disease_Classifier/
├── main.py              # ML pipeline
├── app.py               # Streamlit web app
├── requirements.txt     # Dependencies
├── data/
│   └── heart.csv       # Dataset
├── models/
│   └── model_results.csv  # Results
└── src/
    ├── preprocess.py    # Data preprocessing
    ├── train.py         # Model training
    ├── evaluate.py      # Evaluation metrics
    └── utils.py         # Helper functions
```

## 📦 Requirements

See `requirements.txt` for all dependencies including:
- pandas, numpy, scikit-learn
- xgboost, streamlit
- matplotlib, seaborn, plotly

## 📈 Performance

- **Accuracy**: 80-95%
- **ROC-AUC**: 0.85-0.95

### Modify Model Parameters
Edit `src/train.py` to adjust:
- Hyperparameters for each model
- Cross-validation folds
- Random state for reproducibility

### Change Data Preprocessing
Edit `src/preprocess.py` to:
- Adjust feature scaling methods
- Change train-test split ratio
- Add or remove features

### Customize Web Interface
Edit `app.py` to:
- Modify input parameters
- Change visualization styles
- Add new features to the dashboard

## 🧪 Testing

To verify the installation:

```bash
# Test data loading
python -c "import pandas as pd; df = pd.read_csv('data/heart.csv'); print(f'Dataset shape: {df.shape}')"

# Test imports
python -c "from src.preprocess import DataPreprocessor; print('Imports successful')"

# Run pipeline
python main.py
```

## 📝 Notes

- The project uses scikit-learn's `train_test_split` for data splitting
- Models are trained with stratified sampling to maintain class distribution
- Feature scaling is applied to improve model performance
- XGBoost typically shows the highest accuracy
- All results are saved for reproducibility

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork or clone the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Test thoroughly
5. Submit a pull request with a clear description

### Contribution Ideas:
- Add more classification algorithms
- Improve EDA visualizations
- Enhance the web interface
- Add cross-validation
- Implement ensemble methods
- Add feature engineering techniques

## 📄 License

This project is open source and available under the MIT License. Feel free to use, modify, and distribute as needed.

## 👨‍💻 Author

Created as a machine learning classification project demonstrating end-to-end ML pipeline implementation with Streamlit deployment.

## 📞 Contact & Support

For questions, issues, or suggestions:
- Review the code comments for implementation details
- Check the main.py script for pipeline execution flow
- Examine app.py for web interface customization

## 🎓 Learning Resources

This project demonstrates:
- Data preprocessing and EDA techniques
- Multiple classification algorithms
- Model evaluation and comparison
- Hyperparameter tuning
- Web application development with Streamlit
- ML pipeline automation
- Model deployment

## 🔮 Future Enhancements

- [ ] Add deep learning models (Neural Networks)
- [ ] Implement feature importance analysis
- [ ] Add SHAP interpretability
- [ ] Create REST API with FastAPI
- [ ] Add user authentication to web app
- [ ] Implement model explainability dashboard
- [ ] Add cross-validation scoring
- [ ] Create model versioning system
- [ ] Add real-time monitoring capabilities
- [ ] Implement automated model retraining

---

**Last Updated**: April 2026

**Status**: ✅ Production Ready

For the latest updates and improvements, refer to the project repository.
