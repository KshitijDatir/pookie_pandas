const mongoose = require('mongoose');
const { Schema } = mongoose;

const BankSchema = new Schema({
    name: { type: String, required: true },
    bankCode: { type: String, required: true, unique: true },
    users: [{ type: Schema.Types.ObjectId, ref: 'User' }],
}, { timestamps: true });

const Bank = mongoose.model('Bank', BankSchema);
module.exports = Bank;