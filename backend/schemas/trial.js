const mongoose = require("mongoose");

const transactionSchema = new mongoose.Schema({
  timestamp: {
    type: Date,
    required: true,
    default: Date.now,
  },
  bank_id: {
    type: String,
    required: true,
  },
  sender_account: {
    type: String,
    required: true,
  },
  receiver_account: {
    type: String,
    required: true,
  },
  amount: {
    type: Number,
    required: true,
  },
  transaction_type: {
    type: String,
    enum: ["withdrawal", "deposit", "transfer", "payment"], // add more types if needed
    required: true,
  },
  merchant_category: {
    type: String,
    default: null,
  },
  location: {
    type: String,
    default: null,
  },
  device_used: {
    type: String,
    default: null,
  },
  is_fraud: {
    type: Boolean,
    default: false,
  },
  time_since_last_transaction: {
    type: Number,
    default: 0, // in hours or whatever unit you use
  },
  spending_deviation_score: {
    type: Number,
    default: 0,
  },
  velocity_score: {
    type: Number,
    default: 0,
  },
  geo_anomaly_score: {
    type: Number,
    default: 0,
  },
  payment_channel: {
    type: String,
    enum: ["card", "netbanking", "upi", "cash", "other"],
    default: "card",
  },
  ip_address: {
    type: String,
    default: null,
  },
  device_hash: {
    type: String,
    default: null,
  },
  processed_for_fl: {
    type: Boolean,
    default: false,
  },
});

// Create model
const Transaction = mongoose.model("Transaction", transactionSchema);

module.exports = Transaction;
