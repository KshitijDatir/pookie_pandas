// routes/transaction.js
const express = require("express");
const router = express.Router();
const fs = require("fs");
const path = require("path");
const User = require("../schemas/user.js");
const Bank = require("../schemas/bank.js");
// const { Bank1Transaction, Bank2Transaction, Bank3Transaction } = require("../non_frauds_first500.json");
const { SBI_Q, HDFC_Q, AXIS_Q, SBI_TXN, HDFC_TXN, AXIS_TXN } = require('../schemas/transaction.js');
const { v4: uuidv4 } = require('uuid');

const fraudsPath = path.join(__dirname, "../data/frauds.json");
let fraudData = []; // cache for fraud data

// Helper to select bank transaction model (legacy function)
function getTransactionModel(bankCode) {
  // Use new bank models instead
  switch (bankCode) {
    case "BANK1":
    case "SBI":
      return SBI_Q;
    case "BANK2":
    case "HDFC":
      return HDFC_Q;
    case "BANK3":
    case "AXIS":
      return AXIS_Q;
    default:
      throw new Error("Invalid bank code");
  }
}

// Load fraud data only when needed
function loadFraudData() {
  if (fraudData.length > 0) return; // already loaded
  try {
    const rawData = fs.readFileSync(fraudsPath, "utf-8");
    fraudData = JSON.parse(rawData);
    console.log(`✅ Fraud data loaded on demand (${fraudData.length} records)`);
  } catch (err) {
    console.error("❌ Error loading frauds.json:", err.message);
  }
}

// Get random fraud record
function getRandomFraud() {
  if (fraudData.length === 0) return null;
  const randomIndex = Math.floor(Math.random() * fraudData.length);
  return fraudData[randomIndex];
}

// Default normal (non-fraud) values
const NOT_FRAUD_VALUES = {
  merchant_category: "other",
  location: "unknown",
  device_used: "mobile",
  is_fraud: null,
  time_since_last_transaction: 0,
  spending_deviation_score: 0,
  velocity_score: 0,
  geo_anomaly_score: 0,
  payment_channel: "other",
  device_hash: "dummy-device-hash",
};

router.post("/transaction", async (req, res) => {
  const {
    senderUserID,
    senderBankCode,
    receiverUserID,
    receiverBankCode,
    amount,
    transaction_type,
    useFraudValues, // true → pick random fraud record
    fillFraudData, // true → load fraud data into memory if not loaded
  } = req.body;

  if (!senderUserID || !senderBankCode || !receiverUserID || !receiverBankCode || !amount || !transaction_type) {
    return res.status(400).json({ message: "Missing required fields" });
  }

  try {
    // Load fraud data only if requested
    if (fillFraudData) {
      loadFraudData();
    }

    // Validate banks
    const senderBank = await Bank.findOne({ bankCode: senderBankCode });
    const receiverBank = await Bank.findOne({ bankCode: receiverBankCode });
    if (!senderBank || !receiverBank)
      return res.status(400).json({ message: "Sender or Receiver bank not found" });

    // Validate users
    const sender = await User.findOne({ userID: senderUserID, bankCode: senderBankCode });
    const receiver = await User.findOne({ userID: receiverUserID, bankCode: receiverBankCode });
    if (!sender || !receiver)
      return res.status(400).json({ message: "Sender or Receiver user not found" });

    const TransactionModel = getTransactionModel(senderBankCode);

    // Choose random fraud or normal values
    const RANDOM_FRAUD_VALUES = useFraudValues ? getRandomFraud() : null;
    const STATIC_VALUES = RANDOM_FRAUD_VALUES || NOT_FRAUD_VALUES;

    // Create transaction
    const transaction = await TransactionModel.create({
      senderUserID: sender._id,
      receiverUserID: receiver._id,
      amount,
      transaction_type,
      sender_account: sender.userID,
      receiver_account: receiver.userID,
      ip_address: req.ip,
      merchant_category: STATIC_VALUES.merchant_category,
      location: STATIC_VALUES.location,
      device_used: STATIC_VALUES.device_used,
      time_since_last_transaction: STATIC_VALUES.time_since_last_transaction,
      spending_deviation_score: STATIC_VALUES.spending_deviation_score,
      velocity_score: STATIC_VALUES.velocity_score,
      geo_anomaly_score: STATIC_VALUES.geo_anomaly_score,
      payment_channel: STATIC_VALUES.payment_channel,
      device_hash: STATIC_VALUES.device_hash,
    });

    res.status(201).json({
      message: "Transaction successful",
      fraudDataLoaded: fillFraudData && fraudData.length > 0,
      transaction,
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Helper function to get the correct bank transaction model
function getBankTransactionModel(bankId) {
  switch (bankId.toUpperCase()) {
    case 'SBI':
      return SBI_TXN;
    case 'HDFC':
      return HDFC_TXN;
    case 'AXIS':
      return AXIS_TXN;
    default:
      throw new Error(`Invalid bank ID: ${bankId}`);
  }
}

// Helper function to generate realistic transaction data
function generateTransactionData(formData) {
  const now = new Date();
  
  // Indian cities for realistic location data
  const indianCities = [
    'Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad',
    'Pune', 'Ahmedabad', 'Surat', 'Jaipur', 'Lucknow', 'Kanpur',
    'Nagpur', 'Indore', 'Thane', 'Bhopal', 'Visakhapatnam', 'Vadodara',
    'Firozabad', 'Ludhiana', 'Rajkot', 'Agra', 'Siliguri', 'Nashik',
    'Faridabad', 'Patiala', 'Meerut', 'Kalyan-Dombivli', 'Vasai-Virar', 'Varanasi'
  ];
  
  return {
    // User provided data
    sender_account: formData.sender_account,
    receiver_account: formData.receiver_account,
    amount: formData.amount,
    transaction_type: formData.transaction_type,
    merchant_category: formData.merchant_category,
    
    // Auto-generated realistic data for ML model
    transaction_id: uuidv4(),
    timestamp: now,
    ip_address: '192.168.' + Math.floor(Math.random() * 256) + '.' + Math.floor(Math.random() * 256),
    device_hash: 'device-' + Math.random().toString(36).substr(2, 12),
    location: indianCities[Math.floor(Math.random() * indianCities.length)], // Random Indian city
    location_lat: parseFloat((Math.random() * 180 - 90).toFixed(6)), // Keep coordinates for ML model
    location_long: parseFloat((Math.random() * 360 - 180).toFixed(6)),
    time_since_last_transaction: Math.floor(Math.random() * 7200), // 0-2 hours in seconds
    spending_deviation_score: parseFloat((Math.random() * 2 - 1).toFixed(3)), // -1 to 1
    velocity_score: parseFloat((Math.random() * 10).toFixed(3)), // 0 to 10
    geo_anomaly_score: parseFloat((Math.random() * 5).toFixed(3)), // 0 to 5
    user_id: 'user_' + Math.random().toString(36).substr(2, 8),
    merchant_id: 'merchant_' + Math.random().toString(36).substr(2, 8),
    account_age_days: Math.floor(Math.random() * 3650), // 0-10 years
    transaction_hour: now.getHours(),
    day_of_week: now.getDay(),
    is_weekend: now.getDay() === 0 || now.getDay() === 6,
    num_transactions_today: Math.floor(Math.random() * 20) + 1,
    avg_transaction_amount: parseFloat((formData.amount * (0.8 + Math.random() * 0.4)).toFixed(2)),
    processed_for_fl: false,
    is_fraud: null // Will be predicted by ML model later
  };
}

// New endpoint for frontend transaction simulator
router.post('/simulate-transaction', async (req, res) => {
  try {
    const { 
      sender_account, 
      receiver_account, 
      amount, 
      transaction_type, 
      merchant_category, 
      bank_id 
    } = req.body;

    // Validate required fields
    if (!sender_account || !receiver_account || !amount || !transaction_type || !merchant_category || !bank_id) {
      return res.status(400).json({ 
        message: 'Missing required fields',
        required: ['sender_account', 'receiver_account', 'amount', 'transaction_type', 'merchant_category', 'bank_id']
      });
    }

    // Validate amount is positive number
    if (isNaN(amount) || amount <= 0) {
      return res.status(400).json({ message: 'Amount must be a positive number' });
    }

    // Validate sender and receiver are different
    if (sender_account === receiver_account) {
      return res.status(400).json({ message: 'Sender and receiver accounts must be different' });
    }

    // Get the appropriate bank transaction model
    const BankTransactionModel = getBankTransactionModel(bank_id);
    
    // Generate complete transaction data
    const transactionData = generateTransactionData({
      sender_account,
      receiver_account,
      amount: parseFloat(amount),
      transaction_type,
      merchant_category
    });

    // Save transaction to the appropriate bank collection
    const newTransaction = new BankTransactionModel(transactionData);
    const savedTransaction = await newTransaction.save();

    // Return success response
    res.status(201).json({
      message: 'Transaction simulated successfully',
      transactionId: transactionData.transaction_id,
      bank: bank_id.toUpperCase(),
      collection: `${bank_id.toLowerCase()}_txns`,
      transaction: {
        id: savedTransaction._id,
        transaction_id: transactionData.transaction_id,
        amount: transactionData.amount,
        type: transactionData.transaction_type,
        timestamp: transactionData.timestamp,
        processed_for_fl: transactionData.processed_for_fl
      }
    });
    
  } catch (error) {
    console.error('Transaction simulation error:', error);
    
    if (error.name === 'ValidationError') {
      return res.status(400).json({ 
        message: 'Validation error', 
        details: error.message 
      });
    }
    
    res.status(500).json({ 
      message: 'Internal server error during transaction simulation',
      error: error.message 
    });
  }
});

module.exports = router;
