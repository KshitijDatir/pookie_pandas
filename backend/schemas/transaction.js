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
const Bank1Transaction = mongoose.model('Bank1Transaction', transactionSchemaTemplate);
const Bank2Transaction = mongoose.model('Bank2Transaction', transactionSchemaTemplate);
const Bank3Transaction = mongoose.model('Bank3Transaction', transactionSchemaTemplate);

module.exports = { Bank1Transaction, Bank2Transaction, Bank3Transaction };
