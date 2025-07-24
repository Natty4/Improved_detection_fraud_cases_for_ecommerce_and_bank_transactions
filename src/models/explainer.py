import numpy as np
import shap
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
import logging

logger = logging.getLogger(__name__)

class SHAPExplainer:
    """Handles model explanation using SHAP."""
    
    def explain_ecommerce(self, model, data):
        """Generate SHAP explanations for e-commerce model."""
        try:
            print("\n📊 Generating SHAP explanations for e-commerce model...")
            
            if model is None:
                logger.error("No model provided for explanation")
                return
            
            # Get preprocessed training data
            preprocessor = model.named_steps['preprocessor']
            X_train = data['processed'].drop(columns=['user_id', 'signup_time', 'purchase_time', 
                                                    'device_id', 'ip_address', 'class'])
            
            # Sample data for faster explanation
            sample_size = min(1000, len(X_train))
            X_sample = X_train.sample(sample_size, random_state=42)
            
            try:
                X_train_preprocessed = preprocessor.transform(X_sample)
            except Exception as e:
                logger.error(f"Error transforming data: {str(e)}")
                return
            
            # Get feature names
            feature_names = self._get_feature_names(preprocessor, X_sample)
            
            # Create explainer
            try:
                explainer = shap.Explainer(model.named_steps['model'])
                shap_values = explainer(X_train_preprocessed)
                
                # Generate plots
                shap.summary_plot(shap_values, X_train_preprocessed, feature_names=feature_names)
                return True
            except Exception as e:
                logger.error(f"SHAP explanation failed: {str(e)}")
                return False
                
        except Exception as e:
            logger.error(f"Error in explain_ecommerce: {str(e)}")
            raise
    
    def explain_credit(self, model, data):
        """Generate SHAP explanations for credit card model."""
        try:
            print("\n💳 Generating SHAP explanations for credit card model...")
            
            if model is None:
                logger.error("No model provided for explanation")
                return
            
            # Get preprocessed training data
            preprocessor = model.named_steps['preprocessor']
            X_train = data['processed'].drop(columns=["Class"])
            
            # Sample data for faster explanation
            sample_size = min(1000, len(X_train))
            X_sample = X_train.sample(sample_size, random_state=42)
            
            try:
                X_train_preprocessed = preprocessor.transform(X_sample)
            except Exception as e:
                logger.error(f"Error transforming data: {str(e)}")
                return
            
            # Create explainer
            try:
                explainer = shap.Explainer(model.named_steps['model'])
                shap_values = explainer(X_train_preprocessed)
                
                # Generate plots
                shap.summary_plot(shap_values, X_train_preprocessed, feature_names=X_train.columns.tolist())
                return True
            except Exception as e:
                logger.error(f"SHAP explanation failed: {str(e)}")
                return False
                
        except Exception as e:
            logger.error(f"Error in explain_credit: {str(e)}")
            raise
    
    def _get_feature_names(self, preprocessor, X_train):
        """Extract feature names after preprocessing."""
        try:
            # Get categorical feature names
            cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out()
            
            # Get numeric features (excluding any that might be in categorical)
            numeric_cols = [col for col in X_train.columns 
                          if col not in preprocessor.named_transformers_['cat'].feature_names_in_]
            
            return list(cat_features) + numeric_cols
        except Exception as e:
            logger.error(f"Error getting feature names: {str(e)}")
            return []