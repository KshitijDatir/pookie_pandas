const mongoose = require('mongoose');
const { Schema } = mongoose;

// -------------------- User Schema --------------------
const UserSchema = new Schema({
    userID: { type: String, required: true, unique: true },
    name: { type: String, required: true },
    email: { type: String },
    phone: { type: String },
    bankID: { type: Schema.Types.ObjectId, ref: 'Bank', required: true },
    balance: { type: Number, default: 0 },
}, { timestamps: true });

const User = mongoose.model('User', UserSchema);
module.exports = User;
