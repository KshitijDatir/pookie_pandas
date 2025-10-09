# 🔄 Continuous Server Testing Guide

## ✅ **Current Setup Status:**
- **Server Configuration**: 1000 rounds (continuous mode)
- **Strict Threshold**: Exactly 5+ transactions required for FL participation
- **Current Database State**: 4 transactions (FL BLOCKED)
- **Server Behavior**: Continues running even with insufficient client data

## 🚀 **Testing Procedure:**

### **Step 1: Start the Continuous Server**
```powershell
# Terminal 1 - Start Server (runs continuously)
python start_server.py
```

**Expected Output:**
```
🏦 Federated Learning for Banking Fraud Detection
Using PyTorch Autoencoder with Flower Framework
♾️  CONTINUOUS MODE: Server runs indefinitely
🔒 STRICT THRESHOLD: Clients need ≥5 transactions for participation
🔄 Server continues running even with insufficient client data

⚙️  Loaded configuration: 1000 rounds, continuous mode enabled
🧠 Base model: [model_path]
⚙️  Preprocessor: [preprocessor_path]
🌐 Server address: localhost:8080
♾️  Mode: Continuous operation (1000 rounds)
👥 Minimum clients: 1 (server waits for any client)
🔒 Threshold: Clients need ≥5 transactions for FL participation

🚀 Initializing server...
```

### **Step 2: Connect Client with Insufficient Data (4 transactions)**
```powershell
# Terminal 2 - Start Client
python start_client.py SBI
```

**Expected Behavior:**
- ✅ Client connects to server successfully
- ❌ Client reports insufficient data (4 < 5 transactions)
- 🔄 Server continues running (doesn't stop)
- ⏳ Server waits for client to have sufficient data

**Expected Client Logs:**
```
🏦 Smart bank client initialized for SBI
📊 Current data entries: 4
🎯 STRICT MINIMUM REQUIRED: 5 transactions (no exceptions)
🚫 FL BLOCKED: Need 1 more transactions
```

**Expected Server Logs:**
```
🔄 FL Round 1 - Aggregating results from 1 clients
❌ Client SBI: Insufficient data (4 < 5 transactions)
🚨 1 client(s) blocked due to insufficient data (< 5 transactions)
⏳ No clients with sufficient data (>= 5 transactions) in this round
🔄 Server continues running, waiting for more transaction data...
```

### **Step 3: Add 5th Transaction (Enable FL Participation)**
```powershell
# Terminal 3 - Add the threshold-crossing transaction
python add_test_data.py SBI 1
```

**Expected Real-Time Response:**
- 🎉 Client instantly detects new transaction via Change Streams
- ✅ Threshold reached (5 transactions)
- 🔄 Next FL round proceeds with training
- 🔄 Server continues running after successful round

**Expected Client Logs:**
```
📝 Detected insert operation
📊 Data count updated: 4 → 5
🎉 STRICT THRESHOLD REACHED! 5 >= 5 transactions
✅ Bank is now STRICTLY ready for federated learning!
```

**Expected Server Logs:**
```
🔄 FL Round 2 - Aggregating results from 1 clients
✅ 1 client(s) have sufficient data - proceeding with FL round
Round 2 aggregation completed:
  Total training examples: [X,XXX]
  Average training loss: [X.XXXXXX]
```

### **Step 4: Continue Adding Data (Server Keeps Running)**
```powershell
# Add more transactions to test continuous operation
python add_test_data.py SBI 3
```

**Expected Behavior:**
- 🔄 Server continues running indefinitely
- ✅ Each FL round completes successfully
- 💾 Global model updates after each round
- 🎯 Server never stops due to completed rounds

## 🎯 **Key Features Verified:**

### ✅ **Continuous Operation:**
- Server runs for 1000 rounds (effectively infinite)
- Server doesn't stop after completing FL rounds
- Server continues waiting for clients indefinitely

### ✅ **Strict Threshold Enforcement:**
- Clients with <5 transactions are blocked from FL participation
- Server shows clear logs about insufficient data clients
- Server continues running even when all clients are blocked

### ✅ **Real-Time Threshold Detection:**
- Clients detect new transactions via MongoDB Change Streams instantly
- FL participation automatically enabled when threshold reached
- No manual restarts needed

### ✅ **Graceful Handling:**
- Server handles insufficient data gracefully
- Clear logging about client states
- Server maintains global model state between rounds

## 🛠️ **Troubleshooting:**

### **If Server Stops Unexpectedly:**
- Check that `num_rounds: 1000` in config
- Verify `min_fit_clients: 1` and `min_evaluate_clients: 0`
- Ensure base model files exist in `trained_models/`

### **If Client Can't Connect:**
- Verify server is running on `localhost:8080`
- Check MongoDB connection string
- Ensure client has proper config for bank ID

### **If Threshold Not Working:**
- Database should show exactly your transaction count
- Client logs should show strict threshold checking
- Add/remove transactions to test threshold crossing

## 📈 **Success Criteria:**
- ✅ Server runs continuously without stopping
- ✅ Server handles clients with <5 transactions gracefully
- ✅ FL proceeds automatically when clients reach 5+ transactions
- ✅ Real-time monitoring detects new transactions instantly
- ✅ Global model updates after each successful FL round
