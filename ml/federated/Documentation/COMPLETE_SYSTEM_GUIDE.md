# 🏦 Federated Learning for Banking Fraud Detection

## Complete System Guide & Documentation

This comprehensive guide covers the entire federated learning system for banking fraud detection using PyTorch Autoencoder and Flower framework.

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [System Architecture](#-system-architecture)
3. [File Structure](#-file-structure)
4. [Components Guide](#-components-guide)
5. [Installation & Setup](#-installation--setup)
6. [Usage Guide](#-usage-guide)
7. [Configuration](#-configuration)
8. [API Reference](#-api-reference)
9. [Troubleshooting](#-troubleshooting)
10. [Production Deployment](#-production-deployment)

---

## 🎯 Project Overview

### What This System Does

This federated learning system enables multiple banks to collaboratively train a fraud detection model without sharing their sensitive transaction data. Each bank keeps its data locally while contributing to a global model that benefits from collective knowledge.

### Key Features

- **🔒 Privacy-Preserving**: Bank data never leaves local premises
- **🤝 Collaborative Learning**: Benefits from collective fraud patterns
- **🧠 Autoencoder-Based**: Unsupervised anomaly detection using reconstruction error
- **🌐 Scalable**: Add new banks without system changes
- **📊 Real-time Detection**: Deploy trained models for live fraud detection
- **⚙️ Production-Ready**: Complete error handling, logging, and monitoring

### Technology Stack

- **Deep Learning**: PyTorch
- **Federated Learning**: Flower Framework
- **Database**: MongoDB
- **Data Processing**: pandas, scikit-learn
- **Orchestration**: Custom FedAvg implementation

---

## 🏗 System Architecture

### High-Level Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Bank A (SBI)  │    │   Bank B (HDFC) │    │   Bank C (AXIS) │
│                 │    │                 │    │                 │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │ MongoDB     │ │    │ │ MongoDB     │ │    │ │ MongoDB     │ │
│ │ Transactions│ │    │ │ Transactions│ │    │ │ Transactions│ │
│ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │FL Client    │ │    │ │FL Client    │ │    │ │FL Client    │ │
│ │(Autoencoder)│ │    │ │(Autoencoder)│ │    │ │(Autoencoder)│ │
│ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────▼───────────┐
                    │  Federated Server      │
                    │                        │
                    │ ┌────────────────────┐ │
                    │ │ FedAvg Strategy    │ │
                    │ │ (Parameter Agg.)   │ │
                    │ └────────────────────┘ │
                    │ ┌────────────────────┐ │
                    │ │ Global Model       │ │
                    │ │ (Base Autoencoder) │ │
                    │ └────────────────────┘ │
                    └────────────────────────┘
```

### Data Flow

1. **Initialization**: Server loads pre-trained base model
2. **Distribution**: Server sends initial model parameters to all banks
3. **Local Training**: Each bank trains on their private transaction data
4. **Aggregation**: Server collects and aggregates model updates using FedAvg
5. **Update**: Server updates global model and redistributes to banks
6. **Iteration**: Process repeats for configured number of rounds

### Model Architecture

**Autoencoder (Anomaly Detection)**
```
Input (15 features)
    ↓
Encoder: 15 → 64 → 32 → 16 → 8 (bottleneck)
    ↓
Decoder: 8 → 16 → 32 → 64 → 15
    ↓
Output (15 reconstructed features)
```

**Fraud Detection Logic**:
- Normal transactions: Low reconstruction error
- Fraudulent transactions: High reconstruction error
- Threshold-based classification

---

## 📁 File Structure

```
federated/
├── 📄 README.md                       # Project overview
├── 📄 requirements.txt                # Python dependencies
├── 🚀 train_base_model.py             # Main launcher: Train base model
├── 🚀 start_server.py                 # Main launcher: Start FL server
├── 🚀 start_client.py                 # Main launcher: Start bank client
├── 
├── 🏦 bank_client.py                  # Bank Flower client implementation
├── 🌐 initialize_federated_server.py  # Federated server with FedAvg
├── 📊 flower_server.py                # Alternative server (legacy)
├── 🔄 run_banking_fl.py               # FL orchestration script
│
├── 📁 models/                         # Model definitions
│   └── 🧠 autoencoder.py              # PyTorch Autoencoder model
│
├── 📁 utils/                          # Utility modules
│   └── 🔧 data_preprocessing.py       # Data preprocessing pipeline
│
├── 📁 scripts/                        # Training and utility scripts
│   └── 🏋️ train_base_model.py         # Base model training script
│
├── 📁 config/                         # Configuration files
│   └── ⚙️ federated_config.py         # Central configuration
│
├── 📁 tests/                          # Testing and verification
│   ├── 🧪 test_federated_system.py    # System component tests
│   └── ✅ final_verification.py       # Complete system verification
│
├── 📁 docs/                           # Documentation
│   ├── 📖 COMPLETE_SYSTEM_GUIDE.md    # This file
│   ├── 📚 API_REFERENCE.md            # API documentation
│   └── 🚀 DEPLOYMENT_GUIDE.md         # Production deployment guide
│
└── 📁 trained_models/                 # Saved models and preprocessors
    ├── 🎯 latest_base_model.pth       # Latest trained base model
    ├── ⚙️ latest_preprocessor.pkl     # Fitted data preprocessor
    ├── 📋 base_training.log           # Training logs
    └── 📈 training_history.png        # Training visualization
```

---

## 🧩 Components Guide

### 1. 🧠 Autoencoder Model (`models/autoencoder.py`)

**Purpose**: Neural network for fraud detection using reconstruction error

**Key Features**:
- Symmetric encoder-decoder architecture
- Dropout layers for regularization
- Sigmoid output for normalized data
- Separate encode/decode methods

**Usage**:
```python
from models.autoencoder import Autoencoder

# Create model
model = Autoencoder(input_dim=15)

# Forward pass
output = model(input_tensor)  # Reconstruction

# Get latent representation
encoded = model.encode(input_tensor)

# Reconstruct from latent
decoded = model.decode(encoded)
```

### 2. 🔧 Data Preprocessing (`utils/data_preprocessing.py`)

**Purpose**: Transform raw transaction data into model-ready features

**Key Features**:
- Numerical feature scaling (MinMaxScaler)
- Categorical encoding (label encoding)
- Hash-based feature processing
- Configurable column handling
- Single transaction processing

**Pipeline**:
1. Drop unnecessary columns
2. Encode categorical features
3. Scale numerical features
4. Hash string features
5. Handle missing values
6. Convert data types

**Usage**:
```python
from utils.data_preprocessing import FraudDataPreprocessor

# Initialize and fit
preprocessor = FraudDataPreprocessor()
preprocessor.fit(training_data)

# Transform data
features = preprocessor.transform(new_data)

# Process single transaction
features = preprocessor.extract_features_from_transaction(transaction_dict)
```

### 3. 🌐 Federated Server (`initialize_federated_server.py`)

**Purpose**: Orchestrate federated learning using FedAvg strategy

**Key Components**:
- `FraudDetectionStrategy`: Custom Flower strategy
- Model parameter aggregation
- Client coordination
- Global model management

**FedAvg Implementation**:
- **Aggregation Rule**: `θ_global = Σ(n_k/n_total * θ_k)`
- **Weighting**: Based on client sample counts
- **Updates**: Parameter-level averaging

**Usage**:
```python
# Start server
python start_server.py

# Or programmatically
from initialize_federated_server import start_federated_server
start_federated_server(
    server_address="[::]:8080",
    num_rounds=10,
    min_clients=2
)
```

### 4. 🏦 Bank Client (`bank_client.py`)

**Purpose**: Bank-side federated learning client

**Key Features**:
- MongoDB integration for transaction data
- Local model training
- Dynamic model initialization
- Parameter synchronization with server
- Fraud detection capabilities

**Client Lifecycle**:
1. **Initialization**: Create client with bank configuration
2. **Connection**: Receive initial model parameters from server
3. **Data Collection**: Fetch recent transactions from MongoDB
4. **Local Training**: Train Autoencoder on normal transactions
5. **Parameter Sharing**: Send updated parameters to server
6. **Update**: Receive aggregated global parameters

**Usage**:
```python
# Start client for SBI
python start_client.py SBI

# Start client for HDFC
python start_client.py HDFC

# Or programmatically
from bank_client import AutoencoderBankClient
client = AutoencoderBankClient('SBI', mongo_config)
```

### 5. 🏋️ Base Model Training (`scripts/train_base_model.py`)

**Purpose**: Train initial base model on CSV dataset

**Features**:
- Load and preprocess CSV data
- Train Autoencoder with early stopping
- Save model and preprocessor
- Generate training visualizations
- Comprehensive logging

**Process**:
1. Load `non_frauds_10k.csv` dataset
2. Apply preprocessing pipeline
3. Train Autoencoder (unsupervised)
4. Implement early stopping
5. Save best model and preprocessor

**Usage**:
```python
# Train base model
python train_base_model.py
```

### 6. ⚙️ Configuration (`config/federated_config.py`)

**Purpose**: Central configuration management

**Configuration Categories**:
- **Server**: Address, rounds, client requirements
- **Model**: Architecture, hyperparameters
- **Training**: Epochs, patience, optimization
- **Data**: Paths, preprocessing settings
- **Banks**: MongoDB configurations
- **Security**: TLS, authentication (production)

**Usage**:
```python
from config.federated_config import get_bank_config, get_server_config

# Get bank configuration
bank_config = get_bank_config('SBI')

# Get server configuration
server_config = get_server_config()
```

---

## 🛠 Installation & Setup

### Prerequisites

- **Python**: 3.8 or higher
- **MongoDB**: 4.4 or higher (for production with real data)
- **CUDA**: Optional, for GPU acceleration

### Installation Steps

1. **Clone Repository**
   ```bash
   git clone <your-repo-url>
   cd federated_learning
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Installation**
   ```bash
   python tests/test_federated_system.py
   ```

### MongoDB Setup (Optional)

For production deployment with real bank data:

1. **Install MongoDB**
   ```bash
   # Ubuntu
   sudo apt install mongodb
   
   # Windows: Download from https://www.mongodb.com/try/download/community
   ```

2. **Start MongoDB Service**
   ```bash
   sudo systemctl start mongod
   ```

3. **Create Bank Databases**
   ```javascript
   // Connect to MongoDB
   use sbi_banking_data
   db.sbi_transactions.createIndex({"timestamp": 1, "bank_id": 1})
   
   use hdfc_banking_data
   db.hdfc_transactions.createIndex({"timestamp": 1, "bank_id": 1})
   ```

---

## 🚀 Usage Guide

### Quick Start (3 Steps)

#### Step 1: Train Base Model
```bash
python train_base_model.py
```
**Output**: Trained model saved to `trained_models/`

#### Step 2: Start Federated Server
```bash
python start_server.py
```
**Output**: Server listening on `[::]:8080`

#### Step 3: Connect Bank Clients
```bash
# Terminal 1 - SBI Bank
python start_client.py SBI

# Terminal 2 - HDFC Bank  
python start_client.py HDFC
```
**Output**: Federated learning rounds begin automatically

### Advanced Usage

#### Custom Bank Configuration
```python
# In config/federated_config.py
BANK_CONFIGS["CUSTOM_BANK"] = {
    "bank_id": "CUSTOM_BANK",
    "mongo_config": {
        "connection_string": "mongodb://your-server:27017/",
        "database": "custom_bank_data",
        "collection_template": "{bank_id}_transactions"
    }
}
```

#### Different Server Settings
```bash
# Modify config/federated_config.py
SERVER_CONFIG = {
    "server_address": "[::]:9090",  # Different port
    "num_rounds": 20,               # More rounds
    "min_clients": 3,               # Require 3 banks
}
```

#### Training with GPU
```bash
# Automatic GPU detection
python train_base_model.py  # Will use CUDA if available

# Force CPU usage
CUDA_VISIBLE_DEVICES="" python train_base_model.py
```

### Data Format Requirements

#### MongoDB Transaction Schema
```json
{
  "_id": "ObjectId",
  "timestamp": "ISODate",
  "bank_id": "string",
  "sender_account": "string", 
  "receiver_account": "string",
  "amount": "number",
  "transaction_type": "string",
  "merchant_category": "string", 
  "location": "string",
  "device_used": "string",
  "is_fraud": "boolean",
  "fraud_type": "string",
  "time_since_last_transaction": "number",
  "spending_deviation_score": "number",
  "velocity_score": "number", 
  "geo_anomaly_score": "number",
  "payment_channel": "string",
  "ip_address": "string",
  "device_hash": "string",
  "processed_for_fl": "boolean"
}
```

#### CSV Training Data Schema
Your `non_frauds_10k.csv` should contain the same fields as above.

---

## ⚙️ Configuration

### Server Configuration
```python
SERVER_CONFIG = {
    "server_address": "[::]:8080",     # Server listening address
    "num_rounds": 10,                  # FL training rounds
    "min_clients": 2,                  # Minimum participating banks
    "timeout": 300,                    # Client timeout (seconds)
}
```

### Model Configuration  
```python
MODEL_CONFIG = {
    "input_dim": 15,                   # Features (auto-detected)
    "learning_rate": 0.001,            # Adam optimizer LR
    "batch_size": 128,                 # Training batch size
    "local_epochs": 5,                 # Local epochs per FL round
}
```

### Bank Configuration
```python
# Example for SBI Bank
BANK_CONFIGS["SBI"] = {
    "bank_id": "SBI",
    "mongo_config": {
        "connection_string": "mongodb://localhost:27017/",
        "database": "sbi_banking_data", 
        "collection_template": "sbi_transactions"
    }
}
```

### Fraud Detection Configuration
```python
FRAUD_DETECTION_CONFIG = {
    "default_threshold": 0.1,          # Reconstruction error threshold
    "adaptive_threshold": True,        # Use adaptive thresholding  
    "threshold_percentile": 95,        # Percentile for threshold
}
```

---

## 📚 API Reference

### Core Classes

#### `Autoencoder`
```python
class Autoencoder(nn.Module):
    def __init__(self, input_dim=14)
    def forward(self, x) -> torch.Tensor
    def encode(self, x) -> torch.Tensor  
    def decode(self, z) -> torch.Tensor
```

#### `FraudDataPreprocessor`
```python
class FraudDataPreprocessor:
    def fit(self, df: pd.DataFrame) -> 'FraudDataPreprocessor'
    def transform(self, df: pd.DataFrame) -> np.ndarray
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray
    def extract_features_from_transaction(self, transaction: Dict) -> List[float]
```

#### `AutoencoderBankClient`
```python
class AutoencoderBankClient(fl.client.NumPyClient):
    def __init__(self, bank_id: str, mongo_config: Dict)
    def fit(self, parameters, config) -> Tuple[List, int, Dict]
    def evaluate(self, parameters, config) -> Tuple[float, int, Dict] 
    def detect_fraud(self, transaction_data: Dict) -> Tuple[bool, float]
```

#### `FraudDetectionStrategy`
```python
class FraudDetectionStrategy(FedAvg):
    def __init__(self, base_model_path: str, preprocessor_path: str)
    def initialize_parameters(self, client_manager) -> Parameters
    def aggregate_fit(self, server_round: int, results: List[FitRes]) -> Tuple
    def aggregate_evaluate(self, server_round: int, results: List[EvaluateRes]) -> Tuple
```

### Key Functions

#### Training
```python
def train_base_model(dataset_path: str, model_save_dir: str) -> Dict
def start_federated_server(server_address: str, num_rounds: int) -> None
def start_bank_client(bank_id: str, mongo_config: Dict) -> None
```

#### Configuration
```python 
def get_bank_config(bank_id: str) -> Dict
def get_server_config() -> Dict
def get_model_config() -> Dict
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. Model Dimension Mismatch
**Problem**: `RuntimeError: size mismatch`  
**Solution**: The system auto-detects dimensions. Ensure preprocessor is fitted consistently.

#### 2. MongoDB Connection Failed
**Problem**: `ConnectionError: Failed to connect to MongoDB`
**Solutions**:
- Check MongoDB is running: `systemctl status mongod`
- Verify connection string in configuration
- Check network connectivity

#### 3. No Training Data Available
**Problem**: `WARNING: No new data found for bank`
**Solutions**: 
- Verify MongoDB collection exists and has data
- Check timestamp filters in data collection
- Ensure `processed_for_fl` flag is correctly managed

#### 4. Server/Client Connection Issues
**Problem**: Clients can't connect to server
**Solutions**:
- Check server is running and listening on correct port
- Verify server address configuration
- Check firewall settings
- Test network connectivity: `telnet server_ip 8080`

#### 5. Memory Issues During Training  
**Problem**: `RuntimeError: CUDA out of memory` or system memory issues
**Solutions**:
- Reduce batch size in configuration
- Use CPU instead of GPU: `CUDA_VISIBLE_DEVICES=""`
- Reduce data collection window (`hours_back`)
- Limit samples per round (`max_samples`)

#### 6. Import Errors
**Problem**: `ModuleNotFoundError`
**Solutions**:
- Ensure all dependencies installed: `pip install -r requirements.txt`
- Check Python path includes project directories
- Verify file paths in import statements

### Debugging Commands

#### Test System Components
```bash
# Test all components
python tests/test_federated_system.py

# Verify complete system
python tests/final_verification.py
```

#### Check Model Status
```python
import torch
checkpoint = torch.load('trained_models/latest_base_model.pth')
print(f"Model input dim: {checkpoint['model_config']['input_dim']}")
print(f"Training history: {checkpoint['training_history'].keys()}")
```

#### Verify Data Preprocessing
```python
from utils.data_preprocessing import FraudDataPreprocessor
import pandas as pd

preprocessor = FraudDataPreprocessor()
preprocessor.load_preprocessor('trained_models/latest_preprocessor.pkl')
print(f"Fitted: {preprocessor.is_fitted}")
print(f"Categories: {preprocessor.category_mappings.keys()}")
```

### Log Analysis

#### Server Logs
```bash
# Look for these patterns
grep "Aggregating fit results" logs/server.log    # Successful FL rounds
grep "ERROR" logs/server.log                      # Server errors
grep "Client.*connected" logs/server.log          # Client connections
```

#### Client Logs  
```bash
# Look for these patterns
grep "Training completed" logs/bank_*.log         # Successful training
grep "No new data found" logs/bank_*.log          # Data collection issues
grep "Model initialized" logs/bank_*.log          # Parameter updates
```

---

## 🚀 Production Deployment

### Infrastructure Requirements

#### Server Requirements
- **CPU**: 4+ cores recommended
- **RAM**: 8GB+ (depends on model size and client count)
- **Storage**: 50GB+ (for logs, models, checkpoints)
- **Network**: High-bandwidth connection to all bank clients

#### Bank Client Requirements  
- **CPU**: 2+ cores per client
- **RAM**: 4GB+ per client
- **Storage**: 20GB+ (for local models and data)
- **Database**: MongoDB with adequate storage for transactions

### Security Configuration

#### Enable TLS Encryption
```python
# In config/federated_config.py
SECURITY_CONFIG = {
    "enable_tls": True,
    "cert_path": "/path/to/server.crt", 
    "key_path": "/path/to/server.key",
    "client_auth": True,
}
```

#### MongoDB Security
```javascript
// Enable authentication
use admin
db.createUser({
  user: "flAdmin",
  pwd: "securePassword",
  roles: ["readWriteAnyDatabase"]
})
```

#### Network Security
- Use VPN for bank-server connections
- Configure firewall rules
- Monitor network traffic
- Use secure connection strings

### Monitoring and Logging

#### Comprehensive Logging Setup
```python
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_dir": "logs",
    "max_log_size": "100MB",
    "backup_count": 10,
}
```

#### Key Metrics to Monitor
- **FL Rounds**: Completion rate, duration
- **Client Participation**: Active clients per round
- **Model Performance**: Loss trends, convergence
- **Data Quality**: Sample counts, preprocessing success
- **System Resources**: CPU, memory, disk usage
- **Network**: Connection stability, bandwidth usage

#### Alerting
```bash
# Example monitoring script
#!/bin/bash
LOG_FILE="logs/server.log"
ERROR_COUNT=$(grep -c "ERROR" "$LOG_FILE")
if [ "$ERROR_COUNT" -gt 10 ]; then
    echo "High error count detected: $ERROR_COUNT" | mail -s "FL System Alert" admin@bank.com
fi
```

### Backup and Recovery

#### Model Backups
```bash
# Automated backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="backups/$DATE"
mkdir -p "$BACKUP_DIR"

cp trained_models/*.pth "$BACKUP_DIR/"
cp trained_models/*.pkl "$BACKUP_DIR/" 
cp -r logs "$BACKUP_DIR/"

# Compress backup
tar -czf "$BACKUP_DIR.tar.gz" "$BACKUP_DIR"
rm -rf "$BACKUP_DIR"
```

#### Database Backups  
```bash
# MongoDB backup
mongodump --db sbi_banking_data --out backups/mongodb_$(date +%Y%m%d)
```

### Performance Optimization

#### Server Optimization
- Use powerful CPU for parameter aggregation
- Optimize batch sizes based on available memory
- Configure appropriate client timeouts
- Use connection pooling for multiple clients

#### Client Optimization  
- Implement data sampling for large datasets
- Use efficient MongoDB queries with indexes
- Configure appropriate local training epochs
- Monitor local resource usage

#### Network Optimization
- Use compression for parameter transfer
- Implement efficient serialization
- Configure appropriate timeouts
- Use persistent connections where possible

### Scaling Considerations

#### Horizontal Scaling
- Deploy multiple server instances with load balancing
- Use distributed parameter aggregation
- Implement client sharding by geographic region

#### Adding New Banks
1. Create bank configuration in `config/federated_config.py`
2. Set up MongoDB database for new bank
3. Configure network connectivity
4. Deploy client software at bank premises
5. Test connection and initial training round

### Maintenance Procedures

#### Regular Maintenance Tasks
- **Weekly**: Review logs, check system health
- **Monthly**: Update model checkpoints, clean old logs
- **Quarterly**: Review and update configurations
- **Yearly**: Security audit, dependency updates

#### Update Procedures
1. **Test Updates**: Always test in staging environment
2. **Rolling Updates**: Update clients first, then server  
3. **Model Updates**: Coordinate global model updates
4. **Rollback Plan**: Maintain previous version for rollback

### Compliance and Auditing

#### Regulatory Compliance
- Ensure data privacy regulations compliance (GDPR, PCI-DSS)
- Implement audit trails for all operations
- Document data processing and model decisions
- Regular security assessments

#### Audit Trail
```python
# Example audit logging
import logging
audit_logger = logging.getLogger('audit')
audit_logger.info(f"FL Round {round_num}: {len(clients)} clients participated")
audit_logger.info(f"Model updated with {total_samples} samples")
audit_logger.info(f"Average loss: {avg_loss:.6f}")
```

---

## 🎯 Conclusion

This federated learning system provides a complete, production-ready solution for collaborative banking fraud detection while preserving data privacy. The system is designed to be:

- **Scalable**: Easy to add new banks and handle increasing data volumes
- **Secure**: Bank data never leaves local premises
- **Reliable**: Comprehensive error handling and monitoring
- **Maintainable**: Well-documented, configurable, and testable

For additional support or questions, refer to the troubleshooting section or contact the development team.

---

**📧 Support**: Create an issue in the project repository  
**📖 Documentation**: Check `/docs/` folder for additional guides  
**🔄 Updates**: Watch repository for latest improvements

**🎉 Happy Federated Learning!** 🏦🤖
