"""Main orchestration script for Heart Disease ML pipeline."""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.preprocess import DataPreprocessor
from src.train import ModelTrainer
from src.evaluate import ModelEvaluator
from src.utils import save_model, print_section


def main():
    """Execute the complete ML pipeline."""
    # Configuration
    DATA_PATH = 'data/heart.csv'
    MODEL_SAVE_PATH = 'models/best_model.pkl'

    print_section("HEART DISEASE CLASSIFICATION ML PIPELINE")

    # Step 1: Load and Preprocess Data
    print_section("STEP 1: DATA LOADING & PREPROCESSING")

    preprocessor = DataPreprocessor()

    try:
        preprocessor.load_data(DATA_PATH)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # Step 2: Run EDA
    print_section("STEP 2: EXPLORATORY DATA ANALYSIS")
    preprocessor.run_eda()

    # Step 3: Preprocess and split data
    X_train, X_test, y_train, y_test = preprocessor.preprocess()

    # Step 4: Train all models
    print_section("STEP 3: MODEL TRAINING")

    trainer = ModelTrainer()
    best_models = trainer.train_all_models(X_train, y_train)

    # Step 5: Evaluate all models
    print_section("STEP 4: MODEL EVALUATION")

    evaluator = ModelEvaluator()
    results_df = evaluator.compute_metrics(best_models, X_test, y_test)

    # Step 6: Print comparison table
    print_section("STEP 5: MODEL COMPARISON TABLE")
    print(results_df.to_string(index=False))

    # Step 7: Select best model
    print_section("STEP 6: BEST MODEL SELECTION")
    best_model_name = evaluator.select_best_model(results_df)

    # Step 8: Plot results
    print_section("STEP 7: VISUALIZATIONS")

    # Plot ROC curves
    evaluator.plot_roc_curves(best_models, X_test, y_test)

    # Plot confusion matrix for best model
    best_model = best_models[best_model_name]
    evaluator.plot_confusion_matrix(best_model, X_test, y_test)

    # Step 9: Save best model
    print_section("STEP 8: SAVING BEST MODEL")

    try:
        save_model(best_model, MODEL_SAVE_PATH)
        print(f"✅ Best model saved to {MODEL_SAVE_PATH}")
    except Exception as e:
        print(f"❌ Error saving model: {e}")
        return

    print_section("PIPELINE COMPLETED SUCCESSFULLY")
    print(f"🎉 Best performing model: {best_model_name}")
    print(f"💾 Model saved at: {MODEL_SAVE_PATH}")
    print(f"🚀 Run 'streamlit run app.py' to launch the prediction dashboard")


if __name__ == "__main__":
    main()
