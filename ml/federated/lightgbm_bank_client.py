"""
LightGBM Bank Client for Federated Learning

Smart bank client using LightGBM for fraud detection with real-time monitoring.
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
from datetime import datetime, timedelta
import lightgbm as lgb
import pickle

# Add project paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'models'))
sys.path.append(os.path.join(current_dir, 'utils'))
sys.path.append(os.path.join(current_dir, 'config'))

from models.lightgbm_model import FederatedLightGBM
from utils.lightgbm_preprocessor import LightGBMPreprocessor

class LightGBMBankClient(fl.client.NumPyClient):
    """
    Smart bank client using LightGBM with real-time monitoring.
    
    Features:
    - Real-time MongoDB Change Stream monitoring
    - Automatic FL participation when minimum data threshold is met
    - LightGBM model for fraud detection
    - Dynamic data detection without manual restart
    """
    
    def __init__(self, bank_id: str, mongo_config: Dict):
        self.bank_id = bank_id
        self.mongo_config = mongo_config
        self.logger = self._setup_logging()
        
        # LightGBM model and preprocessor
        self.model = FederatedLightGBM(
            n_estimators=50,  # Reduced for faster federated rounds
            learning_rate=0.1,
            max_depth=6,
            num_leaves=31,
            random_state=42
        )
        self.preprocessor = LightGBMPreprocessor()
        
        # MongoDB connections
        self.mongo_client = None
        self.db = None
        self.collection = None
        self.change_stream = None
        
        # Monitoring components
        self.monitor_thread = None
        self.monitoring_active = False
        
        # Data tracking - STRICT threshold enforcement
        self.current_data_count = 0
        self.min_data_entries = 5  # STRICT: exactly 5 or more required
        self.data_sufficient = False
        self.ready_for_training = False
        self.quiet_mode = os.getenv('FL_QUIET_MODE', 'false').lower() == 'true'
        
        # Load configuration and connect
        self.load_config()
        self.connect_to_mongodb()
        self.start_monitoring()
        
        self.logger.info(f"🏦 LightGBM client initialized: {bank_id} ({self.current_data_count} transactions)")
        if self.current_data_count < 5:
            self.logger.warning(f"🚫 FL BLOCKED: Need {5 - self.current_data_count} more transactions")
        else:
            self.logger.info(f"✅ FL READY: {self.current_data_count} >= 5 transactions")
    
    def _setup_logging(self):
        """Setup logging for this bank."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(levelname)s:%(name)s: %(message)s',
            datefmt='%H:%M:%S'
        )
        logger = logging.getLogger(f"LightGBM_{self.bank_id}")
        # Set pymongo to WARNING to reduce connection noise
        logging.getLogger("pymongo").setLevel(logging.WARNING)
        return logger
    
    def load_config(self):
        """Load configuration settings."""
        try:
            from config.federated_config import get_data_config
            data_config = get_data_config()
            self.min_data_entries = data_config.get("min_data_entries", 5)
        except Exception as e:
            self.logger.warning(f"Could not load config: {e}. Using defaults.")
            self.min_data_entries = 5
    
    def connect_to_mongodb(self):
        """Connect to MongoDB and setup collection."""
        try:
            self.mongo_client = MongoClient(self.mongo_config["connection_string"])
            self.db = self.mongo_client[self.mongo_config["database"]]
            
            # Get collection name
            collection_name = self.mongo_config.get("collection_template", "trial")
            if "{bank_id}" in collection_name:
                collection_name = collection_name.format(bank_id=self.bank_id.lower())
            
            self.collection = self.db[collection_name]
            
            # Get initial data count
            self.current_data_count = self.get_available_data_count()
            self.data_sufficient = self.current_data_count >= 5  # STRICT: hardcoded to 5
            
            self.logger.info(f"✅ Connected to MongoDB ({collection_name})")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to connect to MongoDB: {e}")
            raise
    
    def get_available_data_count(self) -> int:
        """Get count of available data entries for this bank."""
        try:
            query = {
                "bank_id": self.bank_id,
                "processed_for_fl": {"$ne": True}
            }
            count = self.collection.count_documents(query)
            return count
        except Exception as e:
            self.logger.error(f"Failed to count documents: {e}")
            return 0
    
    def start_monitoring(self):
        """Start real-time MongoDB monitoring using Change Streams."""
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_changes, daemon=True)
        self.monitor_thread.start()
        self.logger.info("🔍 Real-time monitoring active")
    
    def _monitor_changes(self):
        """Monitor MongoDB changes using Change Streams."""
        try:
            # Create change stream for insert operations
            pipeline = [
                {
                    '$match': {
                        'operationType': {'$in': ['insert', 'update', 'delete']},
                        'fullDocument.bank_id': self.bank_id
                    }
                }
            ]
            
            self.change_stream = self.collection.watch(pipeline)
            # Change stream ready - monitoring silently in background
            
            while self.monitoring_active:
                try:
                    # Wait for changes with timeout
                    change = self.change_stream.try_next()
                    
                    if change is not None:
                        self._handle_data_change(change)
                    
                    # Reduced sleep to avoid busy waiting
                    time.sleep(2)
                    
                except (pymongo.errors.PyMongoError, pymongo.errors.CursorNotFound) as e:
                    if "interrupted" in str(e).lower() or "cursor" in str(e).lower():
                        self.logger.info("🔄 MongoDB connection interrupted - reconnecting...")
                    else:
                        self.logger.warning(f"Change stream error: {e}. Retrying...")
                    
                    time.sleep(5)
                    # Reconnect change stream
                    try:
                        if self.change_stream:
                            self.change_stream.close()
                        # Shorter timeout for reconnection
                        self.change_stream = self.collection.watch(pipeline, max_await_time_ms=1000)
                    except Exception as reconnect_error:
                        self.logger.warning(f"Reconnection failed: {reconnect_error}")
                        time.sleep(10)
                        
        except Exception as e:
            self.logger.error(f"Monitoring failed: {e}")
            # Fallback to polling
            self._fallback_polling()
    
    def _fallback_polling(self):
        """Fallback polling mechanism when Change Streams fail."""
        self.logger.info("🔄 Falling back to periodic polling - monitoring in background")
        
        last_logged_count = self.current_data_count
        
        while self.monitoring_active:
            try:
                new_count = self.get_available_data_count()
                
                if new_count != self.current_data_count:
                    self.logger.info(f"📊 Data count changed: {self.current_data_count} → {new_count}")
                    self.current_data_count = new_count
                    last_logged_count = new_count
                    
                    # STRICT threshold check: must have exactly 5+ transactions
                    was_sufficient = self.data_sufficient
                    self.data_sufficient = self.current_data_count >= 5  # STRICT: hardcoded to 5
                    
                    if not was_sufficient and self.data_sufficient:
                        self.logger.info(f"🎉 STRICT THRESHOLD REACHED! {self.current_data_count} >= 5 transactions")
                        self.logger.info("✅ Bank is now STRICTLY ready for federated learning!")
                        self.ready_for_training = True
                    elif was_sufficient and not self.data_sufficient:
                        self.logger.warning(f"⚠️ STRICT: Below threshold: {self.current_data_count} < 5 transactions")
                        self.logger.warning("🚫 FL participation BLOCKED")
                        self.ready_for_training = False
                
                time.sleep(30)  # Poll every 30 seconds to reduce noise
                
            except Exception as e:
                self.logger.error(f"Polling error: {e}")
                time.sleep(60)
    
    def _handle_data_change(self, change):
        """Handle detected MongoDB changes."""
        operation = change['operationType']
        
        # Update data count
        new_count = self.get_available_data_count()
        
        if new_count != self.current_data_count:
            old_count = self.current_data_count
            self.current_data_count = new_count
            
            if not self.quiet_mode:
                self.logger.info(f"📝 {operation.capitalize()} detected: Data count {old_count} → {new_count}")
            
            # STRICT threshold crossing check: must have exactly 5+ transactions
            was_sufficient = self.data_sufficient
            self.data_sufficient = self.current_data_count >= 5  # STRICT: hardcoded to 5
            
            if not was_sufficient and self.data_sufficient:
                self.logger.info(f"🎉 THRESHOLD REACHED! {self.current_data_count} >= 5 transactions")
                self.logger.info("✅ FL participation ENABLED")
                self.ready_for_training = True
            elif was_sufficient and not self.data_sufficient:
                self.logger.warning(f"⚠️ Below threshold: {self.current_data_count} < 5 transactions")
                self.logger.warning("🚫 FL participation BLOCKED")
                self.ready_for_training = False
    
    def collect_training_data(self) -> Tuple[pd.DataFrame, int, List]:
        """Collect training data with real-time count and transaction IDs."""
        # Update current data count
        self.current_data_count = self.get_available_data_count()
        
        if self.current_data_count == 0:
            self.logger.warning("No data found")
            return pd.DataFrame(), 0, []
        
        try:
            # Get transactions for training
            query = {
                "bank_id": self.bank_id,
                "processed_for_fl": {"$ne": True}
            }
            
            transactions = list(self.collection.find(query).limit(10000))
            
            if not transactions:
                return pd.DataFrame(), 0, []
            
            # Extract transaction IDs before converting to DataFrame
            transaction_ids = [tx['_id'] for tx in transactions if '_id' in tx]
            
            # Convert to DataFrame
            df = pd.DataFrame(transactions)
            
            # Remove MongoDB ObjectId from DataFrame but keep IDs for deletion
            if '_id' in df.columns:
                df = df.drop(columns=['_id'])
            
            self.logger.info(f"Collected {len(df)} samples for training")
            return df, len(df), transaction_ids
            
        except Exception as e:
            self.logger.error(f"Data collection failed: {e}")
            return pd.DataFrame(), 0, []
    
    def delete_processed_data(self, transaction_ids: List[str]) -> int:
        """Delete processed transactions from MongoDB after model update."""
        try:
            # Delete documents that were used for training
            delete_query = {
                "bank_id": self.bank_id,
                "_id": {"$in": transaction_ids}
            }
            
            result = self.collection.delete_many(delete_query)
            deleted_count = result.deleted_count
            
            if deleted_count > 0:
                self.logger.info(f"🗑️ Deleted {deleted_count} processed transactions from MongoDB")
                # Update our local count
                self.current_data_count = self.get_available_data_count()
            else:
                self.logger.warning("No transactions were deleted")
                
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Failed to delete processed data: {e}")
            return 0
    
    def get_parameters(self, config):
        """Return model parameters."""
        try:
            model_params = self.model.get_model_params()
            if model_params is None:
                self.logger.warning("No model parameters available")
                return [np.array([0.0], dtype=np.float32)]
            
            # Serialize model parameters properly
            import json
            try:
                # Create a comprehensive model state
                model_state = {
                    'model_dump': model_params.get('model_dump', ''),
                    'params': model_params.get('params', {}),
                    'feature_names': model_params.get('feature_names'),
                    'n_features': model_params.get('n_features'),
                    'total_samples': model_params.get('total_samples', 0),
                    'num_trees': model_params.get('num_trees', 0),
                    'model_type': 'lightgbm'
                }
                
                # Convert to JSON and then to bytes
                json_str = json.dumps(model_state, default=str)
                model_bytes = json_str.encode('utf-8')
                
                # Convert bytes to float32 array for Flower
                byte_array = np.frombuffer(model_bytes, dtype=np.uint8)
                param_array = byte_array.astype(np.float32)
                
                self.logger.debug(f"Encoded model parameters: {len(param_array)} values")
                return [param_array]
                
            except Exception as e:
                self.logger.warning(f"Failed to encode model parameters: {e}")
                return [np.array([0.0], dtype=np.float32)]
            
        except Exception as e:
            self.logger.warning(f"Failed to get parameters: {e}")
            return [np.array([0.0], dtype=np.float32)]
    
    def set_parameters(self, parameters):
        """Set model parameters."""
        if not parameters or len(parameters) == 0:
            return
        
        try:
            # Decode parameters back to model
            if len(parameters[0]) > 1:  # Check if we have actual model data
                # Convert float32 array back to bytes
                byte_array = parameters[0].astype(np.uint8)
                model_bytes = byte_array.tobytes()
                
                # Decode JSON string
                import json
                json_str = model_bytes.decode('utf-8', errors='ignore')
                
                if json_str and len(json_str) > 10:  # Basic validation
                    try:
                        # Parse JSON to get model state
                        model_state = json.loads(json_str)
                        
                        if model_state.get('model_type') == 'lightgbm':
                            self.model.set_model_params(model_state)
                            self.logger.info("✅ Model parameters updated from server")
                        else:
                            self.logger.warning("Received non-LightGBM model parameters")
                    except json.JSONDecodeError as je:
                        self.logger.warning(f"Failed to parse JSON model state: {je}")
                        
        except Exception as e:
            self.logger.warning(f"Failed to set model parameters: {e}")
    
    def fit(self, parameters, config):
        """Train model on local data with real-time threshold checking."""
        self.logger.info(f"🔄 FL round started for {self.bank_id}")
        
        # Set received parameters
        self.set_parameters(parameters)
        
        # Real-time data count check - STRICT enforcement
        current_count = self.get_available_data_count()
        
        # STRICT threshold check: must have EXACTLY 5 or more transactions
        if current_count < 5:  # Hardcoded to ensure strict enforcement
            self.logger.warning(f"❌ Insufficient data: {current_count} < 5 transactions required")
            return self.get_parameters(config), 0, {"status": "insufficient_data", "data_count": current_count, "required": 5}
        
        self.logger.info(f"✅ Threshold passed: {current_count} transactions - proceeding with FL")
        
        # Collect and train
        df, num_samples, transaction_ids = self.collect_training_data()
        
        if num_samples == 0 or df.empty:
            return self.get_parameters(config), 0, {"status": "no_data"}
        
        try:
            # Fit preprocessor if not already fitted
            if not self.preprocessor.is_fitted:
                # Create a sample for fitting with required columns
                if 'is_fraud' not in df.columns:
                    df['is_fraud'] = 0  # Add dummy target for preprocessing
                
                self.preprocessor.fit(df)
                self.logger.info("Fitted LightGBM preprocessor on bank data")
            
            # Preprocess data
            processed_df = self.preprocessor.transform(df)
            
            # Separate features and target
            if 'is_fraud' in processed_df.columns:
                X = processed_df.drop(columns=['is_fraud'])
                y = processed_df['is_fraud']
            else:
                # If no fraud labels, create dummy labels for training
                X = processed_df
                y = pd.Series([0] * len(X))  # All normal transactions
            
            # Remove non-numeric columns
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            X = X[numeric_cols]
            
            self.logger.info(f"Training with {len(X)} samples, {X.shape[1]} features")
            
            # Train the model
            self.model.fit(X, y)
            
            # Calculate training metrics
            y_pred = self.model.predict(X)
            accuracy = (y_pred == y).mean()
            
            self.logger.info(f"✅ Training completed: Accuracy={accuracy:.4f}, Samples={num_samples}")
            
            # Delete processed transactions after successful training and model update
            if transaction_ids:
                deleted_count = self.delete_processed_data(transaction_ids)
                self.logger.info(f"🔒 Data privacy: Removed {deleted_count} transactions after processing")
            
            return self.get_parameters(config), num_samples, {
                "accuracy": float(accuracy),
                "status": "trained",
                "data_count": num_samples,
                "deleted_transactions": len(transaction_ids)
            }
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            return self.get_parameters(config), 0, {"status": "training_failed", "error": str(e)}
    
    def evaluate(self, parameters, config):
        """Evaluate model."""
        self.set_parameters(parameters)
        
        df, num_samples, _ = self.collect_training_data()
        if num_samples == 0 or df.empty:
            return 0.0, 0, {}
        
        try:
            # Preprocess data
            processed_df = self.preprocessor.transform(df)
            
            # Separate features and target
            if 'is_fraud' in processed_df.columns:
                X = processed_df.drop(columns=['is_fraud'])
                y = processed_df['is_fraud']
            else:
                X = processed_df
                y = pd.Series([0] * len(X))
            
            # Remove non-numeric columns
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            X = X[numeric_cols]
            
            # Evaluate
            if self.model.is_fitted:
                y_pred = self.model.predict(X)
                accuracy = (y_pred == y).mean()
                return float(1.0 - accuracy), num_samples, {"eval_accuracy": float(accuracy)}
            else:
                return 1.0, num_samples, {"eval_accuracy": 0.0}
                
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            return 1.0, 0, {}
    
    def stop_monitoring(self):
        """Stop monitoring and clean up."""
        self.monitoring_active = False
        if self.change_stream:
            self.change_stream.close()
        if self.mongo_client:
            self.mongo_client.close()
        self.logger.info("🛑 Monitoring stopped")

def start_lightgbm_bank_client(bank_id: str, mongo_config: Dict, server_address: str = "localhost:8080"):
    """
    Start the LightGBM bank client with real-time monitoring.
    
    Args:
        bank_id: Bank identifier (e.g., "SBI", "HDFC")
        mongo_config: MongoDB configuration
        server_address: Flower server address
    """
    
    print(f"🏦 Starting LightGBM Bank Client for: {bank_id}")
    print("🔍 Real-time MongoDB monitoring enabled")
    print("🎯 Automatic FL participation when data threshold is met")
    print("🔒 STRICT THRESHOLD: Need exactly 5+ transactions for FL participation")
    print("🔄 Client runs continuously - connects to server automatically")
    print("🤖 Using LightGBM for fraud detection")
    print("=" * 60)
    
    # Create client
    client = LightGBMBankClient(bank_id, mongo_config)
    
    try:
        # Start Flower client
        fl.client.start_numpy_client(
            server_address=server_address,
            client=client
        )
    except KeyboardInterrupt:
        print("\n🛑 Client stopped by user")
    except Exception as e:
        print(f"❌ Client failed: {e}")
    finally:
        client.stop_monitoring()

def main():
    """Main function."""
    import sys
    
    # Get bank ID from command line
    bank_id = sys.argv[1] if len(sys.argv) > 1 else "SBI"
    
    # Add paths for imports
    sys.path.append(os.path.join(os.path.dirname(__file__), 'config'))
    
    from config.federated_config import get_bank_config, get_server_config
    
    # Get configurations
    bank_config = get_bank_config(bank_id)
    server_config = get_server_config()
    
    # Start LightGBM client
    start_lightgbm_bank_client(
        bank_id=bank_config["bank_id"],
        mongo_config=bank_config["mongo_config"],
        server_address=server_config["server_address"]
    )

if __name__ == "__main__":
    main()
