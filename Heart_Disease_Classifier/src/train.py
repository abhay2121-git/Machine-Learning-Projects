"""Model training module for Heart Disease ML pipeline."""
import logging
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handles training of multiple ML models with hyperparameter tuning."""

    def __init__(self):
        """Initialize the ModelTrainer with model pipelines and hyperparameters."""
        self.models = {}
        self.best_estimators = {}
        self.param_grids = {
            'Logistic Regression': {
                'model__C': [0.01, 0.1, 1, 10],
                'model__solver': ['lbfgs', 'liblinear']
            },
            'Naive Bayes': {
                'model__var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
            },
            'SVC': {
                'model__C': [0.1, 1, 10],
                'model__kernel': ['rbf', 'linear'],
                'model__gamma': ['scale', 'auto']
            },
            'Decision Tree': {
                'model__max_depth': [3, 5, 7, None],
                'model__min_samples_split': [2, 5, 10],
                'model__min_samples_leaf': [1, 2, 4]
            },
            'Random Forest': {
                'model__n_estimators': [100, 200],
                'model__max_depth': [5, 10, None],
                'model__min_samples_split': [2, 5]
            },
            'XGBoost': {
                'model__n_estimators': [100, 200],
                'model__max_depth': [3, 5],
                'model__learning_rate': [0.01, 0.1]
            }
        }
        self._build_pipelines()

    def _build_pipelines(self):
        """Build sklearn Pipelines for each model."""
        # Logistic Regression with scaler
        self.models['Logistic Regression'] = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(max_iter=1000, random_state=42))
        ])

        # Naive Bayes without scaler
        self.models['Naive Bayes'] = Pipeline([
            ('model', GaussianNB())
        ])

        # SVC with scaler
        self.models['SVC'] = Pipeline([
            ('scaler', StandardScaler()),
            ('model', SVC(probability=True, random_state=42))
        ])

        # Decision Tree with scaler
        self.models['Decision Tree'] = Pipeline([
            ('scaler', StandardScaler()),
            ('model', DecisionTreeClassifier(random_state=42))
        ])

        # Random Forest with scaler
        self.models['Random Forest'] = Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestClassifier(random_state=42))
        ])

        # XGBoost with scaler
        self.models['XGBoost'] = Pipeline([
            ('scaler', StandardScaler()),
            ('model', XGBClassifier(
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=42
            ))
        ])

        logger.info("Model pipelines built successfully")

    def train_all_models(self, X_train, y_train):
        """
        Train all models with hyperparameter tuning using GridSearchCV.

        Args:
            X_train (array-like): Training features.
            y_train (array-like): Training labels.

        Returns:
            dict: Dictionary of best estimators for each model.
        """
        logger.info("Starting model training with hyperparameter tuning...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for name, pipeline in self.models.items():
            logger.info(f"\nTraining {name}...")

            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=self.param_grids[name],
                scoring='roc_auc',
                cv=cv,
                n_jobs=-1,
                verbose=1
            )

            grid_search.fit(X_train, y_train)

            self.best_estimators[name] = grid_search.best_estimator_

            logger.info(f"Best parameters for {name}: {grid_search.best_params_}")
            logger.info(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")

        logger.info("\nAll models trained successfully")
        return self.best_estimators

    def get_best_estimators(self):
        """
        Get the trained best estimators for all models.

        Returns:
            dict: Dictionary of best estimators.
        """
        if not self.best_estimators:
            raise ValueError("Models not trained yet. Call train_all_models() first.")
        return self.best_estimators
