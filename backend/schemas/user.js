const mongoose = require('mongoose');
const { Schema } = mongoose;

const UserSchema = new Schema({
    userID: { type: String, required: true, unique: true },
    name: { type: String, required: true },
    email: { type: String },
    phone: { type: String },
    bankCode: { type: String, required: true }, // <-- using bankCode instead of bankID
    balance: { type: Number, default: 0 },
}, { timestamps: true });

const User = mongoose.model('User', UserSchema);
module.exports = User;
