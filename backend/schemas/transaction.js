const mongoose = require('mongoose');
const { Schema } = mongoose;
const transactionSchemaTemplate = new Schema({
    senderUserID: { type: Schema.Types.ObjectId, ref: 'User', required: true },
    receiverUserID: { type: Schema.Types.ObjectId, ref: 'User', required: true },

    // Numerical columns
    amount: { type: Number, required: true },
    time_since_last_transaction: { type: Number, default: 0 },
    spending_deviation_score: { type: Number, default: 0 },
    velocity_score: { type: Number, default: 0 },
    geo_anomaly_score: { type: Number, default: 0 },

    // Categorical columns
    transaction_type: { type: String, enum: ['debit', 'credit', 'transfer'], required: true },
    merchant_category: { type: String },
    location: { type: String },
    device_used: { type: String },
    payment_channel: { type: String },

    // Hashed / sensitive columns
    sender_account: { type: String, required: true },
    receiver_account: { type: String, required: true },
    ip_address: { type: String },
    device_hash: { type: String },

    // Fraud field
    fraud: { type: Boolean, default: null },  // initially unknown
}, { timestamps: true });

// -------------------- Transaction Models for Each Bank --------------------
const SBI_TXN = mongoose.model('SBI_TXN', transactionSchemaTemplate);
const HDFC_TXN = mongoose.model('HDFC_TXN', transactionSchemaTemplate);
const AXIS_TXN = mongoose.model('AXIS_TXN', transactionSchemaTemplate);
const SBI_Q = mongoose.model('SBI_Q', transactionSchemaTemplate);
const HDFC_Q = mongoose.model('HDFC_Q', transactionSchemaTemplate);
const AXIS_Q = mongoose.model('AXIS_Q', transactionSchemaTemplate);

module.exports = {SBI_TXN, HDFC_TXN, AXIS_TXN,SBI_Q,HDFC_Q,AXIS_Q};
