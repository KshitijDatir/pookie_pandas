"""
Federated Learning Configuration

Central configuration file for all federated learning parameters.
Modify these settings to customize your deployment.
"""

# Server Configuration
SERVER_CONFIG = {
    "server_address": "localhost:8080",  # Server address with port
    "num_rounds": 50,               # Reasonable number for continuous operation
    "min_clients": 1,              # Minimum participating banks required  
    "timeout": 300,                 # Client timeout in seconds
    "continuous_mode": True,        # Run server continuously
    "min_fit_clients": 1,          # Minimum clients needed for training round
    "min_evaluate_clients": 0,     # Minimum clients needed for evaluation (0 = skip evaluation)
    "round_timeout": 120,           # Timeout per round in seconds
}

# Model Configuration - LightGBM (matches original training setup)
MODEL_CONFIG = {
    "model_type": "lightgbm",        # Model type: LightGBM tree-based classifier
    "n_estimators": 200,            # Number of boosting rounds (as in original)
    "learning_rate": 0.05,          # LightGBM learning rate (as in original)
    "max_depth": 10,               # Maximum tree depth (as in original)
    "num_leaves": 31,               # Maximum number of leaves in one tree (as in original)
    "random_state": 42,             # Random seed for reproducibility (as in original)
    "verbose": -1,                  # Suppress LightGBM output (as in original)
    
    # Additional LightGBM parameters for federated learning optimization
    "min_child_samples": 20,        # Minimum number of data samples in a leaf
    "subsample": 0.8,               # Subsample ratio of the training instance
    "colsample_bytree": 0.8,        # Subsample ratio of columns when constructing each tree
    "reg_alpha": 0.1,               # L1 regularization term
    "reg_lambda": 0.1,              # L2 regularization term
    "objective": "binary",          # Binary classification objective
    "metric": "binary_logloss",     # Evaluation metric
    "boosting_type": "gbdt",        # Gradient Boosting Decision Tree
    "feature_fraction": 0.9,        # Feature fraction for training
    "bagging_fraction": 0.8,        # Data fraction for training
    "bagging_freq": 5,              # Frequency of bagging
    
    # Federated learning specific parameters
    "federated_rounds": 50,         # Number of federated learning rounds
    "local_training_rounds": 10,    # Reduced for federated setting
}

# Training Configuration - LightGBM
TRAINING_CONFIG = {
    "train_split": 0.8,            # Train/validation split ratio
    "apply_smote": True,            # Apply SMOTE for class balancing (as in original)
    "smote_random_state": 42,      # SMOTE random state (as in original)
    "stratify": True,               # Use stratified split (as in original)
    "test_size": 0.2,              # Test size for train-test split (as in original)
    "random_state": 42,            # Random state for reproducibility (as in original)
    "save_best": True,              # Save best model during training
    "early_stopping_rounds": 100,   # Early stopping for LightGBM
    "eval_metric": "binary_logloss" # Evaluation metric for early stopping
}

# Data Configuration
DATA_CONFIG = {
    "dataset_path": "D:\\Projects\\Fusion Hackathon\\Dataset\\non_frauds_10k.csv",  # Path to base training data
    "hours_back": 24,               # Hours of data to collect for FL training
    "max_samples": 10000,           # Maximum samples per FL round
    "use_normal_only": True,        # Use only normal transactions for training
    "min_data_entries": 5,          # Minimum data entries required per bank for FL participation
}

# Bank-specific MongoDB Configuration Templates
BANK_CONFIGS = {
    "SBI": {
        "bank_id": "SBI",
        "mongo_config": {
            "connection_string": "mongodb+srv://gurudesai2005_db_user:xz8BtHpTGmtm0XGc@cluster0.zntbebn.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0",
            "database": "pookies",
            "collection_template": "sbi_qs"
        }
    },
    "HDFC": {
        "bank_id": "HDFC", 
        "mongo_config": {
            "connection_string": "mongodb+srv://gurudesai2005_db_user:xz8BtHpTGmtm0XGc@cluster0.zntbebn.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0",
            "database": "pookies",
            "collection_template": "hdfc_qs"
        }
    },
    "AXIS": {
        "bank_id": "AXIS",
        "mongo_config": {
            "connection_string": "mongodb+srv://gurudesai2005_db_user:xz8BtHpTGmtm0XGc@cluster0.zntbebn.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0",
            "database": "pookies",
            "collection_template": "axis_qs"
        }
    }
}

# Preprocessing Configuration
PREPROCESSING_CONFIG = {
    "numerical_cols": [
        "amount", 
        "time_since_last_transaction", 
        "spending_deviation_score",
        "velocity_score", 
        "geo_anomaly_score"
    ],
    "categorical_cols": [
        "transaction_type",
        "merchant_category", 
        "location",
        "device_used", 
        "payment_channel"
    ],
    "hash_cols": [
        "sender_account",
        "receiver_account", 
        "ip_address",
        "device_hash"
    ],
    "drop_cols": [
        "transaction_id", 
        "timestamp", 
        "is_fraud", 
        "fraud_type",
        "bank_id",
        "processed_for_fl",
        "_id"
    ]
}

# Fraud Detection Configuration - LightGBM Probability-based
FRAUD_DETECTION_CONFIG = {
    "default_threshold": 0.5,       # Default probability threshold for fraud classification
    "adaptive_threshold": True,     # Use adaptive thresholding based on data
    "threshold_percentile": 95,     # Percentile for adaptive threshold
    "min_threshold": 0.1,          # Minimum threshold value (probability)
    "max_threshold": 0.9,          # Maximum threshold value (probability)
    "use_probability": True,        # Use probability predictions (not binary)
    "confidence_threshold": 0.8,    # High confidence threshold for definitive fraud
    "uncertainty_range": [0.4, 0.6] # Range for uncertain predictions requiring review
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",                # Logging level (DEBUG, INFO, WARNING, ERROR)
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_dir": "logs",             # Directory for log files
    "max_log_size": "10MB",        # Maximum log file size
    "backup_count": 5,             # Number of backup log files
}

# Security Configuration (for production)
SECURITY_CONFIG = {
    "enable_tls": False,           # Enable TLS encryption (set True for production)
    "cert_path": None,             # Path to TLS certificate
    "key_path": None,              # Path to TLS private key
    "client_auth": False,          # Require client authentication
}

# Paths Configuration - LightGBM Models
PATHS_CONFIG = {
    "trained_models_dir": "trained_models",
    "logs_dir": "logs",
    "checkpoints_dir": "checkpoints",
    "results_dir": "results",
    "base_model_name": "lightgbm_model.pkl",           # Base LightGBM model (original)
    "current_model_name": "latest_lightgbm_federated.pkl", # Current federated model
    "base_preprocessor_name": "latest_preprocessor.pkl",    # Preprocessor for feature engineering
    "versions_dir": "versions",                            # Directory for model versions
    "metadata_file": "model_metadata.json"                 # Version metadata
}

def get_bank_config(bank_id: str) -> dict:
    """
    Get configuration for a specific bank.
    
    Args:
        bank_id: Bank identifier (e.g., "SBI", "HDFC")
        
    Returns:
        Bank-specific configuration dictionary
    """
    if bank_id.upper() in BANK_CONFIGS:
        return BANK_CONFIGS[bank_id.upper()].copy()
    else:
        # Return template for custom bank
        return {
            "bank_id": bank_id,
            "mongo_config": {
                "connection_string": "mongodb+srv://gurudesai2005_db_user:xz8BtHpTGmtm0XGc@cluster0.zntbebn.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0",
                "database": "pookies",
                "collection_template": f"{bank_id.lower()}_qs"
            }
        }

def get_server_config() -> dict:
    """Get server configuration."""
    return SERVER_CONFIG.copy()

def get_model_config() -> dict:
    """Get model configuration."""
    return MODEL_CONFIG.copy()

def get_training_config() -> dict:
    """Get training configuration.""" 
    return TRAINING_CONFIG.copy()

def get_data_config() -> dict:
    """Get data configuration."""
    return DATA_CONFIG.copy()

# Export all configurations for easy importing
__all__ = [
    'SERVER_CONFIG',
    'MODEL_CONFIG', 
    'TRAINING_CONFIG',
    'DATA_CONFIG',
    'BANK_CONFIGS',
    'PREPROCESSING_CONFIG',
    'FRAUD_DETECTION_CONFIG',
    'LOGGING_CONFIG',
    'SECURITY_CONFIG',
    'PATHS_CONFIG',
    'get_bank_config',
    'get_server_config',
    'get_model_config',
    'get_training_config',
    'get_data_config'
]
