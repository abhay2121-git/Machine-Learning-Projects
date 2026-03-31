"""Model evaluation module for Heart Disease ML pipeline."""
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix
)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Handles evaluation of trained ML models."""

    def __init__(self):
        """Initialize the ModelEvaluator."""
        self.results = None

    def compute_metrics(self, models_dict, X_test, y_test):
        """
        Compute evaluation metrics for all models.

        Args:
            models_dict (dict): Dictionary of model name -> model object.
            X_test (array-like): Test features.
            y_test (array-like): Test labels.

        Returns:
            pandas.DataFrame: DataFrame with metrics for each model.
        """
        logger.info("Computing evaluation metrics...")

        metrics_list = []

        for name, model in models_dict.items():
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            metrics = {
                'Model': name,
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred),
                'Recall': recall_score(y_test, y_pred),
                'F1 Score': f1_score(y_test, y_pred),
                'ROC-AUC': roc_auc_score(y_test, y_pred_proba)
            }

            metrics_list.append(metrics)
            logger.info(f"Metrics computed for {name}")

        self.results = pd.DataFrame(metrics_list)

        # Round metrics for display
        numeric_cols = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
        self.results[numeric_cols] = self.results[numeric_cols].round(4)

        return self.results

    def plot_roc_curves(self, models_dict, X_test, y_test, save_path=None):
        """
        Plot ROC curves for all models on a single figure.

        Args:
            models_dict (dict): Dictionary of model name -> model object.
            X_test (array-like): Test features.
            y_test (array-like): Test labels.
            save_path (str, optional): Path to save the figure.

        Returns:
            matplotlib.figure.Figure: The ROC curves figure.
        """
        logger.info("Plotting ROC curves...")

        plt.figure(figsize=(10, 8))

        for name, model in models_dict.items():
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc_score = roc_auc_score(y_test, y_pred_proba)

            plt.plot(
                fpr, tpr,
                label=f"{name} (AUC = {auc_score:.4f})",
                linewidth=2
            )

        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Model Comparison', fontsize=14)
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curves saved to {save_path}")

        plt.tight_layout()
        plt.show()

        return plt.gcf()

    def plot_confusion_matrix(self, best_model, X_test, y_test, save_path=None):
        """
        Plot confusion matrix for the best model using seaborn heatmap.

        Args:
            best_model: Trained model object.
            X_test (array-like): Test features.
            y_test (array-like): Test labels.
            save_path (str, optional): Path to save the figure.

        Returns:
            matplotlib.figure.Figure: The confusion matrix figure.
        """
        logger.info("Plotting confusion matrix...")

        y_pred = best_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            cbar=True,
            xticklabels=['No Disease', 'Heart Disease'],
            yticklabels=['No Disease', 'Heart Disease']
        )
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title('Confusion Matrix - Best Model', fontsize=14)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")

        plt.tight_layout()
        plt.show()

        return plt.gcf()

    def select_best_model(self, results_df):
        """
        Select the best model based on ROC-AUC score.

        Args:
            results_df (pandas.DataFrame): DataFrame with model metrics.

        Returns:
            str: Name of the best model.
        """
        best_idx = results_df['ROC-AUC'].idxmax()
        best_model_name = results_df.loc[best_idx, 'Model']
        best_roc_auc = results_df.loc[best_idx, 'ROC-AUC']

        print("\n" + "=" * 50)
        print(f"🏆 BEST MODEL: {best_model_name}")
        print(f"   ROC-AUC Score: {best_roc_auc:.4f}")
        print("=" * 50 + "\n")

        logger.info(f"Best model selected: {best_model_name} (ROC-AUC: {best_roc_auc:.4f})")

        return best_model_name
