import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from config.settings import DATA_PATHS

class DataProcessor:
    """Handles all data loading, cleaning, and feature engineering."""
    
    def __init__(self):
        self.data_paths = DATA_PATHS
        
    def process_ecommerce_data(self):
        """Process the e-commerce transaction data."""
        # Load data
        fraud_df = pd.read_csv(self.data_paths['fraud_data'])
        ip_df = pd.read_csv(self.data_paths['ip_data'])
        
        # Clean and merge
        df = self._clean_ecommerce_data(fraud_df, ip_df)
        
        # Feature engineering
        df = self._engineer_ecommerce_features(df)
        
        return {
            'raw': fraud_df,
            'processed': df,
            'ip_mapping': ip_df
        }
    
    def process_credit_data(self):
        """Process the credit card transaction data."""
        df = pd.read_csv(self.data_paths['credit_data'])
        return {
            'raw': df,
            'processed': self._clean_credit_data(df)
        }
    
    def _clean_ecommerce_data(self, df, ip_df):
        """Clean e-commerce data."""
        # Type conversion
        df['signup_time'] = pd.to_datetime(df['signup_time'])
        df['purchase_time'] = pd.to_datetime(df['purchase_time'])
        df['ip_address'] = df['ip_address'].astype('int64')
        
        # Remove duplicates and missing values
        df = df.drop_duplicates().dropna(subset=['signup_time', 'purchase_time'])
        
        # Merge country info
        ip_df['lower_bound_ip_address'] = ip_df['lower_bound_ip_address'].astype('int64')
        df['country'] = df['ip_address'].apply(self._find_country, ip_df=ip_df)
        
        return df
    
    def _engineer_ecommerce_features(self, df):
        """Create features for e-commerce data."""
        # Time-based features
        df['hour_of_day'] = df['purchase_time'].dt.hour
        df['day_of_week'] = df['purchase_time'].dt.dayofweek
        df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds() / 3600
        
        # Transaction patterns
        txn_count = df.groupby('user_id').size().rename("transaction_count")
        df = df.merge(txn_count, on='user_id')
        
        df['purchase_date'] = df['purchase_time'].dt.date
        daily_spend = df.groupby(['user_id', 'purchase_date'])['purchase_value'].sum().rename("daily_spend")
        df = df.merge(daily_spend, on=['user_id', 'purchase_date'])
        
        return df.drop(columns=['purchase_date'])
    
    def _clean_credit_data(self, df):
        """Clean credit card data."""
        return df.drop_duplicates()
    
    @staticmethod
    def _find_country(ip, ip_df):
        """Helper to map IP to country."""
        match = ip_df[
            (ip_df['lower_bound_ip_address'] <= ip) & 
            (ip_df['upper_bound_ip_address'] >= ip)
        ]
        return match.iloc[0]['country'] if not match.empty else 'Unknown'
    
    
