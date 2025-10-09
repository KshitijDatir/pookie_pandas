# 🔧 Troubleshooting Guide

## Common Issues and Solutions

### ❌ **Issue 1: Client Cannot Connect to Server (Windows)**

**Error Message:**
```
grpc._channel._MultiThreadedRendezvous: <_MultiThreadedRendezvous of RPC that terminated with:
status = StatusCode.UNAVAILABLE
details = "failed to connect to all addresses; last error: UNAVAILABLE: ipv6:[::]:8080: ConnectEx"
```

**Cause:** IPv6 connectivity issues on Windows

**✅ Solution:**
1. **Stop current server** (press Ctrl+C in server terminal)
2. **Verify configuration** shows `localhost:8080`:
   ```bash
   # Check config/federated_config.py line 10 shows:
   "server_address": "localhost:8080",
   ```
3. **Restart server:**
   ```bash
   python start_server.py
   ```
4. **Connect client:**
   ```bash
   python start_client.py SBI
   ```

---

### ❌ **Issue 2: Port Already in Use**

**Error Message:**
```
OSError: [WinError 10048] Only one usage of each socket address is normally permitted
```

**✅ Solution:**
1. **Find process using port 8080:**
   ```bash
   netstat -ano | findstr :8080
   ```
2. **Kill the process:**
   ```bash
   taskkill /PID <process_id> /F
   ```
3. **Restart server:**
   ```bash
   python start_server.py
   ```

---

### ❌ **Issue 3: MongoDB Connection Failed**

**Error Message:**
```
pymongo.errors.ServerSelectionTimeoutError: connection timeout
```

**✅ Solution:**
1. **Check MongoDB connection string** in `config/federated_config.py`
2. **Test connection manually:**
   ```python
   import pymongo
   client = pymongo.MongoClient("your_connection_string")
   print(client.list_database_names())
   ```
3. **Verify database exists** and has data

---

### ❌ **Issue 4: No Trained Base Model Found**

**Error Message:**
```
FileNotFoundError: No trained base model found
```

**✅ Solution:**
1. **Train base model first:**
   ```bash
   python train_base_model.py
   ```
2. **Verify files exist:**
   ```bash
   dir trained_models\latest_*.* 
   ```

---

### ❌ **Issue 5: Flower Deprecation Warnings**

**Warning Message:**
```
WARNING: DEPRECATED FEATURE: flwr.client.start_numpy_client() is deprecated
```

**✅ Solution:**
This is just a warning about future Flower versions. The system works correctly.

**To suppress warnings:**
```bash
set PYTHONWARNINGS=ignore::DeprecationWarning
python start_client.py SBI
```

---

## 🛠️ **Quick Diagnostic Commands**

### Check System Status:
```bash
# Check if server is running
netstat -ano | findstr :8080

# Check trained models exist
dir trained_models

# Test MongoDB connection (SBI bank)
python -c "import pymongo; client = pymongo.MongoClient('mongodb+srv://gurudesai2005_db_user:xz8BtHpTGmtm0XGc@cluster0.zntbebn.mongodb.net/'); print('✅ MongoDB connected')"

# Check Python imports work
python -c "from bank_client import AutoencoderBankClient; print('✅ Imports work')"
```

### Clean Restart:
```bash
# Kill any existing processes
taskkill /F /IM python.exe

# Clean start
python start_server.py
# (in new terminal)
python start_client.py SBI
```

---

## 🎯 **Expected Working Output**

### Server Output:
```
🌟 Starting Federated Learning Server for Fraud Detection
============================================================
🧠 Base model: D:\...\trained_models\latest_base_model.pth
⚙️ Preprocessor: D:\...\trained_models\latest_preprocessor.pkl
🌐 Server address: localhost:8080
INFO: Flower ECE: gRPC server running (10 rounds)
INFO: [ROUND 1]
```

### Client Output:
```
🏦 Starting bank client for: SBI
🧠 Using PyTorch Autoencoder for fraud detection
INFO: Connected to server
INFO: Model initialized with input_dim=15
INFO: Starting local training...
```

---

## 📞 **Still Having Issues?**

1. **Check all file paths** are correct in configuration
2. **Verify Python environment** has all required packages
3. **Try running system tests:**
   ```bash
   python tests/final_verification.py
   ```
4. **Check detailed logs** in the `logs/` directory (if enabled)

---

*For more help, refer to [QUICK_START.md](QUICK_START.md) or [SYSTEM_SUMMARY.md](SYSTEM_SUMMARY.md)*
