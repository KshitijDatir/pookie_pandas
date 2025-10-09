"""
LightGBM Data Preprocessor for Federated Learning

Handles data preprocessing specifically optimized for LightGBM models
in a federated learning environment.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import pickle
import logging
from typing import Dict, List, Optional, Tuple

class LightGBMPreprocessor:
    """
    Data preprocessor optimized for LightGBM in federated learning.
    Handles categorical encoding, missing values, and feature engineering.
    """
    
    def __init__(self):
        """Initialize the preprocessor."""
        self.is_fitted = False
        
        # Encoders for categorical variables
        self.label_encoders = {}
        self.categorical_cols = []
        self.numerical_cols = []
        
        # Imputers
        self.numerical_imputer = SimpleImputer(strategy='median')
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        
        # Column information
        self.feature_names = []
        self.n_features = None
        self.target_col = 'is_fraud'
        
        # Preprocessing statistics
        self.preprocessing_stats = {}
        
    def fit(self, df: pd.DataFrame):
        """
        Fit the preprocessor on training data.
        
        Args:
            df: Training dataframe
        """
        if self.target_col not in df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in dataframe")
        
        # Separate features from target
        X = df.drop(columns=[self.target_col])
        
        # Columns to skip/drop during preprocessing
        columns_to_skip = {
            'timestamp',  # Will be processed separately
            '_id',        # MongoDB ID
            'processed_for_fl',  # FL processing flag
            'bank_id',    # Bank identifier (keep for federated context but don't use as feature)
            'transaction_time',  # Duplicate timestamp
            'created_at'  # System timestamp
        }
        
        # Start with predefined categorical columns
        predefined_categorical = [
            'transaction_type', 'merchant_category', 'location', 
            'device_used', 'payment_channel', 'fraud_type',
            'sender_account', 'receiver_account', 'ip_address', 'device_hash'
        ]
        
        # Auto-detect categorical columns by checking data types and content
        self.categorical_cols = []
        self.numerical_cols = []
        
        for col in X.columns:
            # Skip certain columns
            if col in columns_to_skip:
                continue
            
            # Check if column is in predefined categorical list
            if col in predefined_categorical and col in X.columns:
                self.categorical_cols.append(col)
                continue
            
            # Check column data type and content
            # Skip datetime columns - they'll be handled by timestamp processing
            if pd.api.types.is_datetime64_any_dtype(X[col]):
                continue  # Skip - will be processed by _process_timestamp
            
            try:
                # Try to convert to numeric
                pd.to_numeric(X[col], errors='raise')
                # If successful, it's numerical
                self.numerical_cols.append(col)
            except (ValueError, TypeError):
                # If conversion fails, it's categorical
                self.categorical_cols.append(col)
        
        logging.info(f"Identified {len(self.categorical_cols)} categorical columns")
        logging.info(f"Identified {len(self.numerical_cols)} numerical columns")
        
        # Handle timestamp if present
        if 'timestamp' in X.columns:
            X = self._process_timestamp(X)
        
        # Fit imputers for numerical columns  
        if self.numerical_cols:
            # Filter out any datetime columns that might have been misclassified
            valid_numerical_cols = []
            for col in self.numerical_cols:
                if col in X.columns and not pd.api.types.is_datetime64_any_dtype(X[col]):
                    valid_numerical_cols.append(col)
            
            if valid_numerical_cols:
                numerical_data = X[valid_numerical_cols]
                self.numerical_imputer.fit(numerical_data)
                # Update the numerical_cols to only include valid ones
                self.numerical_cols = valid_numerical_cols
        
        # Fit label encoders for categorical columns
        for col in self.categorical_cols:
            if col in X.columns:
                # Handle missing values first
                col_data = X[col].fillna('Unknown')
                
                le = LabelEncoder()
                le.fit(col_data)
                self.label_encoders[col] = le
        
        # Store feature names after preprocessing
        processed_X = self._transform_features(X)
        if not hasattr(self, 'feature_names') or not self.feature_names:
            self.feature_names = processed_X.columns.tolist()
            self.n_features = len(self.feature_names)
            
        # Ensure target column is properly typed for classification
        if self.target_col in df.columns:
            df[self.target_col] = df[self.target_col].astype(int)
        
        # Calculate preprocessing statistics
        self._calculate_stats(df)
        
        self.is_fitted = True
        logging.info(f"Preprocessor fitted with {self.n_features} features")
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted preprocessor.
        
        Args:
            df: Dataframe to transform
            
        Returns:
            Processed dataframe
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        # Separate features from target if present
        has_target = self.target_col in df.columns
        if has_target:
            X = df.drop(columns=[self.target_col])
            y = df[self.target_col]
        else:
            X = df.copy()
        
        # Process features
        X_processed = self._transform_features(X)
        
        # Combine with target if it was present
        if has_target:
            # Ensure target is properly typed for classification
            y_processed = y.astype(int)
            X_processed[self.target_col] = y_processed
        
        return X_processed
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(df)
        return self.transform(df)
    
    def _process_timestamp(self, X: pd.DataFrame) -> pd.DataFrame:
        """Extract useful features from timestamp."""
        X = X.copy()
        
        if 'timestamp' in X.columns:
            # Convert to datetime if not already
            X['timestamp'] = pd.to_datetime(X['timestamp'], errors='coerce')
            
            # Extract time-based features
            X['hour'] = X['timestamp'].dt.hour
            X['day_of_week'] = X['timestamp'].dt.dayofweek
            X['month'] = X['timestamp'].dt.month
            X['is_weekend'] = (X['timestamp'].dt.dayofweek >= 5).astype(int)
            
            # Add to numerical columns
            time_features = ['hour', 'day_of_week', 'month', 'is_weekend']
            self.numerical_cols.extend([col for col in time_features if col not in self.numerical_cols])
            
            # Drop original timestamp
            X = X.drop(columns=['timestamp'])
        
        return X
    
    def _transform_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply all transformations to features."""
        X = X.copy()
        
        # Process timestamp if present
        if 'timestamp' in X.columns:
            X = self._process_timestamp(X)
        
        # Handle missing values in numerical columns
        if self.numerical_cols:
            # Filter out any datetime columns that might have been misclassified
            valid_numerical_cols = []
            for col in self.numerical_cols:
                if col in X.columns and not pd.api.types.is_datetime64_any_dtype(X[col]):
                    valid_numerical_cols.append(col)
            
            if valid_numerical_cols:
                numerical_data = X[valid_numerical_cols]
                X[valid_numerical_cols] = self.numerical_imputer.transform(numerical_data)
        
        # Encode categorical columns
        for col in self.categorical_cols:
            if col in X.columns:
                # Handle missing values
                col_data = X[col].fillna('Unknown')
                
                # Handle unseen categories
                le = self.label_encoders[col]
                unique_values = set(le.classes_)
                
                # Map unseen categories to 'Unknown' if it exists, else to most frequent class
                def safe_transform(val):
                    if val in unique_values:
                        return le.transform([val])[0]
                    elif 'Unknown' in unique_values:
                        return le.transform(['Unknown'])[0]
                    else:
                        return le.transform([le.classes_[0]])[0]
                
                X[col] = col_data.apply(safe_transform)
        
        # Remove columns we explicitly want to skip (but keep processed features)
        columns_to_drop = ['_id', 'processed_for_fl', 'bank_id', 'transaction_time', 'created_at']
        for col in columns_to_drop:
            if col in X.columns:
                X = X.drop(columns=[col])
        
        # If we have fitted feature names, try to match them
        if hasattr(self, 'feature_names') and self.feature_names:
            # Get intersection of available columns and expected features
            available_cols = [col for col in self.feature_names if col in X.columns]
            
            # Only filter if we have matching columns, otherwise keep all
            if available_cols:
                X = X[available_cols]
                
                # Add missing columns if any (fill with 0)
                for col in self.feature_names:
                    if col not in X.columns:
                        X[col] = 0
                
                # Ensure correct order
                X = X[self.feature_names]
        
        # Update feature names if this is during fitting
        if not hasattr(self, 'feature_names') or not self.feature_names:
            self.feature_names = X.columns.tolist()
            self.n_features = len(self.feature_names)
        
        return X
    
    def _calculate_stats(self, df: pd.DataFrame):
        """Calculate preprocessing statistics."""
        self.preprocessing_stats = {
            'total_samples': len(df),
            'n_features': self.n_features,
            'categorical_columns': self.categorical_cols.copy(),
            'numerical_columns': self.numerical_cols.copy(),
            'missing_values': df.isnull().sum().to_dict(),
            'fraud_rate': df[self.target_col].mean() if self.target_col in df.columns else None
        }
    
    def get_feature_names(self) -> List[str]:
        """Get feature names after preprocessing."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted first")
        return self.feature_names.copy()
    
    def save_preprocessor(self, filepath: str):
        """Save preprocessor to file."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before saving")
        
        preprocessor_data = {
            'is_fitted': self.is_fitted,
            'label_encoders': self.label_encoders,
            'categorical_cols': self.categorical_cols,
            'numerical_cols': self.numerical_cols,
            'numerical_imputer': self.numerical_imputer,
            'categorical_imputer': self.categorical_imputer,
            'feature_names': self.feature_names,
            'n_features': self.n_features,
            'target_col': self.target_col,
            'preprocessing_stats': self.preprocessing_stats
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(preprocessor_data, f)
        
        logging.info(f"Preprocessor saved to {filepath}")
    
    def load_preprocessor(self, filepath: str) -> bool:
        """Load preprocessor from file."""
        try:
            with open(filepath, 'rb') as f:
                preprocessor_data = pickle.load(f)
            
            # Restore all attributes
            for key, value in preprocessor_data.items():
                setattr(self, key, value)
            
            logging.info(f"Preprocessor loaded from {filepath}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to load preprocessor: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """Get preprocessing statistics."""
        return self.preprocessing_stats.copy()
    
    def preprocess_for_federated_learning(self, df: pd.DataFrame, bank_id: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Preprocess data specifically for federated learning.
        
        Args:
            df: Raw dataframe
            bank_id: Bank identifier
            
        Returns:
            Processed dataframe and metadata
        """
        # Add bank_id if not present
        if 'bank_id' not in df.columns:
            df['bank_id'] = bank_id
        
        # Transform the data
        processed_df = self.transform(df)
        
        # Calculate metadata for federated learning
        metadata = {
            'bank_id': bank_id,
            'n_samples': len(processed_df),
            'n_features': self.n_features,
            'fraud_rate': processed_df[self.target_col].mean() if self.target_col in processed_df.columns else None,
            'feature_names': self.feature_names
        }
        
        return processed_df, metadata
