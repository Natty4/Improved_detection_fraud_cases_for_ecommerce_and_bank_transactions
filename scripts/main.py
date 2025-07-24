
import sys, pathlib
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
        print("\n🔧 Processing e-commerce data...")
        ecom_data = data_processor.process_ecommerce_data()
        
        # 2. Process credit card data
        print("\n💳 Processing credit card data...")
        credit_data = data_processor.process_credit_data()
        
        # 3. Train models
        print("\n🤖 Training models...")
        ecom_models = model_trainer.train_ecommerce_models(ecom_data)
        credit_models = model_trainer.train_credit_models(credit_data)
        
        # 4. Generate explanations
        print("\n🔍 Generating explanations...")
        if 'xgb' in ecom_models.get('models', {}):
            print("\nE-commerce Model Explanation:")
            if not explainer.explain_ecommerce(ecom_models['models']['xgb'], ecom_data):
                print("Failed to generate e-commerce explanations")
        else:
            print("No XGBoost model available for e-commerce data")
            
        if 'xgb' in credit_models.get('models', {}):
            print("\nCredit Card Model Explanation:")
            if not explainer.explain_credit(credit_models['models']['xgb'], credit_data):
                print("Failed to generate credit card explanations")
        else:
            print("No XGBoost model available for credit card data")
        
        print("\n✅ Pipeline completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()