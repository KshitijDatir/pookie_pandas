#!/usr/bin/env python3
"""
Test LightGBM Client - Simplified Version

A simpler version of the LightGBM client for initial testing and debugging.
"""

import os
import sys
import time
import threading
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import pymongo
from pymongo import MongoClient
import flwr as fl
import lightgbm as lgb
import pickle

# Add project paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'config'))

class SimpleLightGBMClient(fl.client.NumPyClient):
    """Simplified LightGBM client for testing."""
    
    def __init__(self, bank_id: str, mongo_config: Dict):
        self.bank_id = bank_id
        self.mongo_config = mongo_config
        self.logger = self._setup_logging()
        
        # Simple LightGBM model
        self.model = lgb.LGBMClassifier(
            n_estimators=10,  # Very small for testing
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
            verbose=-1
        )
        self.model_fitted = False
        
        # MongoDB connection
        self.mongo_client = None
        self.db = None
        self.collection = None
        
        # Connect
        self.connect_to_mongodb()
        
        self.logger.info(f"🧪 Simple LightGBM client initialized for {bank_id}")
    
    def _setup_logging(self):
        """Setup logging."""
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        return logging.getLogger(f"TestLGB_{self.bank_id}")
    
    def connect_to_mongodb(self):
        """Connect to MongoDB."""
        try:
            self.mongo_client = MongoClient(self.mongo_config["connection_string"])
            self.db = self.mongo_client[self.mongo_config["database"]]
            
            collection_name = self.mongo_config.get("collection_template", "trial")
            if "{bank_id}" in collection_name:
                collection_name = collection_name.format(bank_id=self.bank_id.lower())
            
            self.collection = self.db[collection_name]
            self.logger.info(f"✅ Connected to MongoDB: {collection_name}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to connect to MongoDB: {e}")
            raise
    
    def get_data_count(self) -> int:
        """Get count of available data."""
        try:
            count = self.collection.count_documents({
                "bank_id": self.bank_id,
                "processed_for_fl": {"$ne": True}
            })
            return count
        except Exception as e:
            self.logger.error(f"Failed to count documents: {e}")
            return 0
    
    def collect_data(self) -> Tuple[pd.DataFrame, int]:
        """Collect training data."""
        try:
            query = {"bank_id": self.bank_id, "processed_for_fl": {"$ne": True}}
            transactions = list(self.collection.find(query).limit(1000))
            
            if not transactions:
                return pd.DataFrame(), 0
            
            df = pd.DataFrame(transactions)
            if '_id' in df.columns:
                df = df.drop(columns=['_id'])
            
            self.logger.info(f"Collected {len(df)} samples")
            return df, len(df)
            
        except Exception as e:
            self.logger.error(f"Data collection failed: {e}")
            return pd.DataFrame(), 0
    
    def get_parameters(self, config):
        """Return simple model parameters."""
        if self.model_fitted:
            # Return a simple representation
            return [np.array([1.0, 2.0, 3.0], dtype=np.float32)]
        else:
            return [np.array([0.0], dtype=np.float32)]
    
    def set_parameters(self, parameters):
        """Set model parameters."""
        if parameters and len(parameters) > 0:
            self.logger.info("Received parameters from server")
    
    def fit(self, parameters, config):
        """Train the model."""
        self.logger.info(f"🔄 Training started for {self.bank_id}")
        
        # Set parameters
        self.set_parameters(parameters)
        
        # Check data count
        data_count = self.get_data_count()
        if data_count < 5:
            self.logger.warning(f"❌ Insufficient data: {data_count} < 5")
            return self.get_parameters(config), 0, {"status": "insufficient_data", "data_count": data_count}
        
        # Collect data
        df, num_samples = self.collect_data()
        if num_samples == 0:
            return self.get_parameters(config), 0, {"status": "no_data"}
        
        try:
            # Simple preprocessing - just use numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            X = df[numeric_cols].fillna(0)  # Simple fill
            
            # Create dummy target if not present
            if 'is_fraud' in df.columns:
                y = df['is_fraud'].fillna(0).astype(int)
            else:
                y = np.zeros(len(X))  # All normal transactions
            
            # Keep only first 10 features to avoid issues
            if X.shape[1] > 10:
                X = X.iloc[:, :10]
            
            self.logger.info(f"Training with {len(X)} samples, {X.shape[1]} features")
            
            # Train
            self.model.fit(X, y)
            self.model_fitted = True
            
            # Simple accuracy
            y_pred = self.model.predict(X)
            accuracy = (y_pred == y).mean()
            
            self.logger.info(f"✅ Training completed: Accuracy={accuracy:.4f}")
            
            return self.get_parameters(config), num_samples, {
                "accuracy": float(accuracy),
                "status": "trained"
            }
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            return self.get_parameters(config), 0, {"status": "training_failed"}
    
    def evaluate(self, parameters, config):
        """Evaluate the model."""
        self.set_parameters(parameters)
        
        df, num_samples = self.collect_data()
        if num_samples == 0:
            return 1.0, 0, {}
        
        try:
            # Simple evaluation
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            X = df[numeric_cols].fillna(0)
            
            if 'is_fraud' in df.columns:
                y = df['is_fraud'].fillna(0).astype(int)
            else:
                y = np.zeros(len(X))
            
            if X.shape[1] > 10:
                X = X.iloc[:, :10]
            
            if self.model_fitted:
                y_pred = self.model.predict(X)
                accuracy = (y_pred == y).mean()
                loss = 1.0 - accuracy
                return float(loss), num_samples, {"eval_accuracy": float(accuracy)}
            else:
                return 1.0, num_samples, {"eval_accuracy": 0.0}
                
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            return 1.0, 0, {}

def start_test_client(bank_id: str):
    """Start the test client."""
    print(f"🧪 Starting Test LightGBM Client for: {bank_id}")
    
    try:
        from config.federated_config import get_bank_config, get_server_config
        
        bank_config = get_bank_config(bank_id)
        server_config = get_server_config()
        
        client = SimpleLightGBMClient(bank_id, bank_config["mongo_config"])
        
        # Start client
        fl.client.start_numpy_client(
            server_address=server_config["server_address"],
            client=client
        )
        
    except Exception as e:
        print(f"❌ Test client failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function."""
    bank_id = sys.argv[1] if len(sys.argv) > 1 else "SBI"
    start_test_client(bank_id)

if __name__ == "__main__":
    main()
