// routes/transaction.js
const express = require("express");
const router = express.Router();
const fs = require("fs");
const path = require("path");
const User = require("../schemas/user.js");
const Bank = require("../schemas/bank.js");
const { Bank1Transaction, Bank2Transaction, Bank3Transaction } = require("../schemas/transaction.js");

const fraudsPath = path.join(__dirname, "../data/frauds.json");
let fraudData = []; // cache for fraud data

// Helper to select bank transaction model
function getTransactionModel(bankCode) {
  switch (bankCode) {
    case "BANK1":
      return Bank1Transaction;
    case "BANK2":
      return Bank2Transaction;
    case "BANK3":
      return Bank3Transaction;
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
    const STATIC_VALUES = RANDOM_FRAUD_VALUES || REAL_VALUES;

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

module.exports = router;
