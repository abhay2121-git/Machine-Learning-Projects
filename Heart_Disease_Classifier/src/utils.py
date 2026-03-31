"""Utility functions for Heart Disease ML pipeline."""
import joblib
import logging


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def save_model(model, path):
    """
    Save a trained model to disk using joblib.

    Args:
        model: Trained model object to save.
        path (str): File path where the model will be saved.

    Raises:
        Exception: If saving fails.
    """
    try:
        joblib.dump(model, path)
        logger.info(f"Model saved successfully to {path}")
    except Exception as e:
        logger.error(f"Error saving model to {path}: {str(e)}")
        raise


def load_model(path):
    """
    Load a trained model from disk using joblib.

    Args:
        path (str): File path of the saved model.

    Returns:
        object: Loaded model object.

    Raises:
        FileNotFoundError: If the model file does not exist.
        Exception: If loading fails.
    """
    try:
        model = joblib.load(path)
        logger.info(f"Model loaded successfully from {path}")
        return model
    except FileNotFoundError:
        logger.error(f"Model file not found: {path}")
        raise
    except Exception as e:
        logger.error(f"Error loading model from {path}: {str(e)}")
        raise


def print_section(title):
    """
    Print a formatted section divider for CLI output.

    Args:
        title (str): Section title to display.
    """
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")
