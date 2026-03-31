"""Data preprocessing module for Heart Disease ML pipeline."""
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handles data loading, EDA, and preprocessing for heart disease dataset."""

    def __init__(self):
        """Initialize the DataPreprocessor."""
        self.df = None
        self.X = None
        self.y = None
        self.label_encoders = {}
        self.numeric_columns = None
        self.categorical_columns = None

    def load_data(self, path):
        """
        Load CSV data from the specified path.

        Args:
            path (str): Path to the CSV file.

        Returns:
            pandas.DataFrame: Loaded dataset.

        Raises:
            FileNotFoundError: If the file does not exist.
            Exception: For other loading errors.
        """
        try:
            self.df = pd.read_csv(path)
            logger.info(f"Data loaded successfully from {path}")
            logger.info(f"Dataset shape: {self.df.shape}")
            logger.info(f"\nDataset head:\n{self.df.head()}")
            return self.df
        except FileNotFoundError:
            logger.error(f"File not found: {path}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def run_eda(self):
        """
        Perform exploratory data analysis on the loaded dataset.

        Prints:
            - Missing values per column
            - Class distribution of target variable
            - Data types of each column
            - Basic descriptive statistics
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")

        logger.info("=" * 50)
        logger.info("EXPLORATORY DATA ANALYSIS")
        logger.info("=" * 50)

        # Missing values
        logger.info("\n1. Missing Values:")
        missing = self.df.isnull().sum()
        logger.info(missing[missing > 0] if missing.sum() > 0 else "No missing values")

        # Class distribution
        target_col = self.df.columns[-1]
        logger.info(f"\n2. Class Distribution ({target_col}):")
        logger.info(self.df[target_col].value_counts())
        logger.info(f"Class proportions:\n{self.df[target_col].value_counts(normalize=True)}")

        # Data types
        logger.info("\n3. Data Types:")
        logger.info(self.df.dtypes)

        # Basic statistics
        logger.info("\n4. Descriptive Statistics:")
        logger.info(self.df.describe())

        logger.info("=" * 50)

    def preprocess(self):
        """
        Preprocess the data by handling missing values and encoding categoricals.

        Missing values are handled as:
            - Numeric: filled with median
            - Categorical: filled with mode

        Categorical columns are encoded using LabelEncoder.

        Returns:
            tuple: (X_train, X_test, y_train, y_test) - Split datasets.
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")

        logger.info("Starting preprocessing...")

        df_processed = self.df.copy()

        # Identify numeric and categorical columns
        self.numeric_columns = df_processed.select_dtypes(
            include=[np.number]
        ).columns.tolist()

        # Exclude target from features
        target_col = df_processed.columns[-1]
        if target_col in self.numeric_columns:
            self.numeric_columns.remove(target_col)

        self.categorical_columns = df_processed.select_dtypes(
            include=['object', 'category']
        ).columns.tolist()

        logger.info(f"Numeric columns: {self.numeric_columns}")
        logger.info(f"Categorical columns: {self.categorical_columns}")

        # Handle missing values
        # Numeric: fill with median
        for col in self.numeric_columns:
            if df_processed[col].isnull().sum() > 0:
                median_val = df_processed[col].median()
                df_processed[col].fillna(median_val, inplace=True)
                logger.info(f"Filled missing values in {col} with median: {median_val}")

        # Categorical: fill with mode
        for col in self.categorical_columns:
            if df_processed[col].isnull().sum() > 0:
                mode_val = df_processed[col].mode()[0]
                df_processed[col].fillna(mode_val, inplace=True)
                logger.info(f"Filled missing values in {col} with mode: {mode_val}")

        # Encode categorical columns
        for col in self.categorical_columns:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            self.label_encoders[col] = le
            logger.info(f"Encoded categorical column: {col}")

        # Separate features and target
        self.X = df_processed.iloc[:, :-1]
        self.y = df_processed.iloc[:, -1]

        logger.info(f"Features shape: {self.X.shape}")
        logger.info(f"Target shape: {self.y.shape}")

        # Train-test split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y,
            test_size=0.2,
            random_state=42,
            stratify=self.y
        )

        logger.info(f"Train set size: {len(X_train)}")
        logger.info(f"Test set size: {len(X_test)}")
        logger.info("Preprocessing completed successfully.")

        return X_train, X_test, y_train, y_test

    def get_feature_names(self):
        """
        Get the feature column names.

        Returns:
            list: List of feature column names.
        """
        if self.X is None:
            raise ValueError("Data not preprocessed. Call preprocess() first.")
        return self.X.columns.tolist()
