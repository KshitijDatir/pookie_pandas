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

# Model Configuration
MODEL_CONFIG = {
    "input_dim": 15,                # Input features (auto-detected from base model)
    "architecture": {
        "encoder": [64, 32, 16, 8], # Encoder layer sizes
        "decoder": [16, 32, 64],    # Decoder layer sizes (excluding output)
    },
    "learning_rate": 0.001,         # Adam optimizer learning rate
    "batch_size": 128,              # Training batch size
    "local_epochs": 5,              # Local training epochs per FL round
    "dropout_rate": 0.2,            # Dropout for regularization
}

# Training Configuration
TRAINING_CONFIG = {
    "max_epochs": 100,              # Maximum epochs for base model training
    "patience": 15,                 # Early stopping patience
    "train_split": 0.8,            # Train/validation split ratio
    "criterion": "MSELoss",         # Loss function
    "save_best": True,              # Save best model during training
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

# Fraud Detection Configuration
FRAUD_DETECTION_CONFIG = {
    "default_threshold": 0.1,       # Default reconstruction error threshold
    "adaptive_threshold": True,     # Use adaptive thresholding
    "threshold_percentile": 95,     # Percentile for adaptive threshold
    "min_threshold": 0.01,         # Minimum threshold value
    "max_threshold": 1.0,          # Maximum threshold value
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

# Paths Configuration
PATHS_CONFIG = {
    "trained_models_dir": "trained_models",
    "logs_dir": "logs",
    "checkpoints_dir": "checkpoints",
    "results_dir": "results",
    "base_model_name": "latest_base_model.pth",
    "base_preprocessor_name": "latest_preprocessor.pkl",
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
