const mongoose = require('mongoose');
const { Schema } = mongoose;
const transactionSchemaTemplate = new Schema({
    // Optional legacy fields for existing transactions
    senderUserID: { type: Schema.Types.ObjectId, ref: 'User' },
    receiverUserID: { type: Schema.Types.ObjectId, ref: 'User' },

    // Core transaction fields
    sender_account: { type: String, required: true },
    receiver_account: { type: String, required: true },
    amount: { type: Number, required: true },
    transaction_type: { type: String, required: true },
    merchant_category: { type: String, required: true },
    
    // Auto-generated ML fields
    transaction_id: { type: String, unique: true },
    timestamp: { type: Date, default: Date.now },
    ip_address: { type: String },
    device_hash: { type: String },
    location_lat: { type: Number },
    location_long: { type: Number },
    time_since_last_transaction: { type: Number, default: 0 },
    spending_deviation_score: { type: Number, default: 0 },
    velocity_score: { type: Number, default: 0 },
    geo_anomaly_score: { type: Number, default: 0 },
    user_id: { type: String },
    merchant_id: { type: String },
    account_age_days: { type: Number },
    transaction_hour: { type: Number },
    day_of_week: { type: Number },
    is_weekend: { type: Boolean },
    num_transactions_today: { type: Number },
    avg_transaction_amount: { type: Number },
    processed_for_fl: { type: Boolean, default: false },
    is_fraud: { type: Number, default: null },
    
    // Legacy fields for backward compatibility
    location: { type: String },
    device_used: { type: String },
    payment_channel: { type: String },
    fraud: { type: Boolean, default: null }
}, { timestamps: true });

// -------------------- Transaction Models for Each Bank --------------------
// TXN collections for user input transactions
const SBI_TXN = mongoose.model('sbi_txn', transactionSchemaTemplate, 'sbi_txns');
const HDFC_TXN = mongoose.model('hdfc_txn', transactionSchemaTemplate, 'hdfc_txns');
const AXIS_TXN = mongoose.model('axis_txn', transactionSchemaTemplate, 'axis_txns');

// Q collections for federated learning processed data  
const SBI_Q = mongoose.model('sbi_q', transactionSchemaTemplate, 'sbi_qs');
const HDFC_Q = mongoose.model('hdfc_q', transactionSchemaTemplate, 'hdfc_qs');
const AXIS_Q = mongoose.model('axis_q', transactionSchemaTemplate, 'axis_qs');

module.exports = {SBI_TXN, HDFC_TXN, AXIS_TXN,SBI_Q,HDFC_Q,AXIS_Q};
