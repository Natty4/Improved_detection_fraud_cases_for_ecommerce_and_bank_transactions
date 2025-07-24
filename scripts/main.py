import sys
import pathlib
import joblib  # Using joblib since that's what you used to save models
ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR
sys.path.append(str(SRC_DIR))

from src.core.data_processing import DataProcessor
from src.models.model_training import ModelTrainer
from src.models.explainer import SHAPExplainer
from config.settings import DATA_PATHS

def main():
    # Initialize components
    data_processor = DataProcessor()
    model_trainer = ModelTrainer()
    explainer = SHAPExplainer()
    
    print("🚀 Starting fraud detection pipeline...")
    
    try:
        # 1. Process e-commerce data
        print("\n🛒 Processing e-commerce data...")
        ecom_data = data_processor.process_ecommerce_data()
        
        # 2. Process credit card data
        print("\n💳 Processing credit card data...")
        credit_data = data_processor.process_credit_data()
        
        # 3. Train and save models
        print("\n🤖 Training models...")
        ecom_models = model_trainer.train_ecommerce_models(ecom_data)
        credit_models = model_trainer.train_credit_models(credit_data)
        
        # 4. Explain using saved models
        print("\n🔍 Generating explanations...")
        
        # Option 1: Use the in-memory models from the training results
        if 'xgb' in ecom_models.get('models', {}):
            print("\nE-commerce Model Explanation:")
            explainer.explain_ecommerce(model=ecom_models['models']['xgb'], data=ecom_data)
        else:
            print("No XGBoost model available for e-commerce data")
            
        if 'xgb' in credit_models.get('models', {}):
            print("\nCredit Card Model Explanation:")
            explainer.explain_credit(model=credit_models['models']['xgb'], data=credit_data)
        else:
            print("No XGBoost model available for credit card data")
        
        # Option 2
        if 'model_paths' in ecom_models:
            try:
                ecom_model = joblib.load(ecom_models['model_paths']['xgb'])
                explainer.explain_ecommerce(model=ecom_model, data=ecom_data)
            except Exception as e:
                print(f"Failed to load/explain e-commerce model: {str(e)}")
        
        if 'model_paths' in credit_models:
            try:
                credit_model = joblib.load(credit_models['model_paths']['xgb'])
                explainer.explain_credit(model=credit_model, data=credit_data)
            except Exception as e:
                print(f"Failed to load/explain credit model: {str(e)}")
        
        print("\n✅ Pipeline completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()