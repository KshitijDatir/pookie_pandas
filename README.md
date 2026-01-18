# 🏦 Federated Learning for Banking Fraud Detection

**Privacy-Preserving Collaborative Fraud Detection using Federated Learning**

A production-ready federated learning system that enables multiple banks to collaboratively train a fraud detection model **without sharing sensitive transaction data**.

---

## ✨ Key Highlights

- 🔒 Privacy-preserving — bank data never leaves local infrastructure
- 🧠 PyTorch autoencoder for unsupervised fraud detection
- 🌐 Federated learning using Flower (FedAvg)
- 🗄️ MongoDB integration for real banking transaction data
- ⚙️ Production-ready with logging and error handling

---

## 🧩 System Architecture

Banks (Clients) → Local Training → Model Updates  
Banks (Clients) ← Global Model (FedAvg) ← Server

Each bank trains locally on private transaction data and shares **only model parameters** with the federated server.

---

## 🚀 Quick Start

1) Train Base Model  
python train_base_model.py

2) Start Federated Server  
python initialize_federated_server.py

3) Connect Bank Clients (run separately)  
python bank_client.py SBI  
python bank_client.py HDFC  

Federated learning begins automatically once clients connect.

---

## 📁 Project Structure

federated/  
├── train_base_model.py  
├── initialize_federated_server.py  
├── bank_client.py  
├── data_preprocessing.py  
├── models/  
│   └── autoencoder.py  
├── trained_models/  
│   ├── latest_base_model.pth  
│   ├── latest_preprocessor.pkl  
│   └── training_history.png  
└── README.md  

---

## 🧠 Model Details

Autoencoder Architecture

Input (14 features)  
→ 14 → 64 → 32 → 16 → 8  
→ 8 → 16 → 32 → 64 → 14  
→ Reconstructed Output

- Unsupervised training on normal transactions  
- High reconstruction error indicates potential fraud  

---

## 📊 Features Used (14)

Numerical  
- amount  
- time_since_last_transaction  
- spending_deviation_score  
- velocity_score  
- geo_anomaly_score  

Categorical  
- transaction_type  
- merchant_category  
- location  
- device_used  
- payment_channel  

Hashed Identifiers  
- sender_account  
- receiver_account  
- ip_address  
- device_hash  

---

## 🔄 Federated Learning Workflow

1. Train base model on historical non-fraud data  
2. Server distributes global model to banks  
3. Banks train locally on private data  
4. Server aggregates updates using FedAvg  
5. Updated global model redistributed  
6. Repeat for multiple rounds  

---

## 🗄️ MongoDB Transaction Format

{
  "bank_id": "SBI",
  "sender_account": "ACC877572",
  "receiver_account": "ACC388389",
  "amount": 343.78,
  "transaction_type": "withdrawal",
  "merchant_category": "utilities",
  "location": "Tokyo",
  "device_used": "mobile",
  "payment_channel": "card",
  "ip_address": "13.101.214.112",
  "device_hash": "D8536477",
  "is_fraud": false,
  "processed_for_fl": false
}

---

## 🚨 Fraud Detection Example

is_fraud, error = client.detect_fraud(transaction)

If is_fraud:  
→ Fraud detected (high reconstruction error)  
Else:  
→ Normal transaction  

---

## 📈 Monitoring & Logging

- Server logs federated rounds and aggregation metrics  
- Client logs local training progress  
- Training curves saved as plots  
- Debug logging supported  

---

## 🛠️ Configuration

Federated Server  
num_rounds = 10  
min_clients = 2  
server_address = [::]:8080  

Base Training  
epochs = 100  
patience = 15  

---

## 🧪 Common Issues

- No base model found → Run train_base_model.py  
- MongoDB connection error → Verify connection string  
- Clients not connecting → Ensure server is running  
- Insufficient clients → Start at least 2 banks  

---

## 🔮 Future Improvements

- Secure aggregation  
- Differential privacy  
- Real-time transaction streaming  
- Model versioning  
- Monitoring dashboard  

---

## 📜 License

MIT License

---

## 🙌 Acknowledgements

PyTorch  
Flower Federated Learning  
MongoDB
