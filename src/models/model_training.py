from pathlib import Path
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline  # This is the critical change
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, 
                            roc_auc_score, average_precision_score)

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Handles model training and evaluation."""
    def __init__(self):
        self.model_dir = Path("outputs/models")
        self.model_dir.mkdir(exist_ok=True)
        
    def train_ecommerce_models(self, data):
        """Train models on e-commerce data."""
        try:
            logger.info("Preparing e-commerce data for training...")
            X = data['processed'].drop(columns=['user_id', 'signup_time', 'purchase_time', 
                                            'device_id', 'ip_address', 'class'])
            y = data['processed']['class']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42
            )
            
            # Define preprocessing
            categorical_cols = ['source', 'browser', 'sex', 'country']
            numeric_cols = [col for col in X_train.columns if col not in categorical_cols]
            
            preprocessor = ColumnTransformer([
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols),
                ('num', StandardScaler(), numeric_cols)
            ])
            
            # Train models
            logger.info("Training Logistic Regression...")
            logreg = self._train_logistic_regression(preprocessor, X_train, y_train)
            
            logger.info("Training XGBoost...")
            xgb = self._train_xgboost(preprocessor, X_train, y_train)
            
            # Evaluate
            results = {
                'logreg': self._evaluate_model(logreg, X_test, y_test, "Logistic Regression"),
                'xgb': self._evaluate_model(xgb, X_test, y_test, "XGBoost"),
                'test_data': (X_test, y_test)
            }
            
            # Save models
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            joblib.dump(logreg, self.model_dir / f"credit_logreg_{timestamp}.pkl")
            joblib.dump(xgb, self.model_dir / f"credit_xgb_{timestamp}.pkl")
            
            return {
                'models': {'logreg': logreg, 'xgb': xgb},
                'results': results,
                'model_paths': {
                    'logreg': str(self.model_dir / f"credit_logreg_{timestamp}.pkl"),
                    'xgb': str(self.model_dir / f"credit_xgb_{timestamp}.pkl")
                }
            }
            
        except Exception as e:
            logger.error(f"Error training e-commerce models: {str(e)}")
            raise
    
    def train_credit_models(self, data):
        """Train models on credit card data."""
        X = data['processed'].drop(columns=["Class"])
        y = data['processed']["Class"]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Preprocessing (just scaling for credit data)
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), X_train.columns.tolist())
        ])
        
        # Train models
        logreg = self._train_logistic_regression(preprocessor, X_train, y_train)
        xgb = self._train_xgboost(preprocessor, X_train, y_train)
        
        # Evaluate
        results = {
            'logreg': self._evaluate_model(logreg, X_test, y_test, "Logistic Regression (Credit)"),
            'xgb': self._evaluate_model(xgb, X_test, y_test, "XGBoost (Credit)"),
            'test_data': (X_test, y_test)
        }
        
        return {
            'models': {'logreg': logreg, 'xgb': xgb},
            'results': results
        }
    
    def _train_logistic_regression(self, preprocessor, X_train, y_train):
        """Train logistic regression model."""
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('model', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
        ])
        pipe.fit(X_train, y_train)
        return pipe
    
    def _train_xgboost(self, preprocessor, X_train, y_train):
        """Train XGBoost model."""
        try:
            logger.info("Building XGBoost pipeline...")
            pipe = Pipeline([
                ('preprocessor', preprocessor),
                ('smote', SMOTE(random_state=42)),
                ('model', XGBClassifier(
                    n_estimators=150, 
                    learning_rate=0.1, 
                    max_depth=5,
                    scale_pos_weight=1,
                    eval_metric='logloss',
                    random_state=42
                ))
            ])
            logger.info("Fitting XGBoost model...")
            pipe.fit(X_train, y_train)
            return pipe
        except Exception as e:
            logger.error(f"Error training XGBoost: {str(e)}")
            raise
    
    def _evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate model performance."""
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'pr_auc': average_precision_score(y_test, y_proba)
        }
        
        return metrics