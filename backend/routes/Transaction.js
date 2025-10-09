const express = require('express');
const router = express.Router();
const User = require('../schemas/user.js');
const Bank = require('../schemas/bank.js');
const { Bank1Transaction, Bank2Transaction, Bank3Transaction } = require('../schemas/transaction.js');

// routes/transactions.js


function getTransactionModel(bankCode) {
    switch(bankCode) {
        case 'BANK1': return Bank1Transaction;
        case 'BANK2': return Bank2Transaction;
        case 'BANK3': return Bank3Transaction;
        default: throw new Error('Invalid bank code');
    }
}

// ------------------- LOGIN -------------------
router.post('/login', async (req, res) => {
    const { userID, bankCode } = req.body;
    try {
        const bank = await Bank.findOne({ bankCode });
        if (!bank) return res.status(400).json({ message: 'Bank not found' });

        const user = await User.findOne({ userID, bankID: bank._id });
        if (!user) return res.status(400).json({ message: 'User not found in this bank' });

        res.json({ message: 'Login successful', user });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// ------------------- MAKE TRANSACTION -------------------
router.post('/transaction', async (req, res) => {
    const { senderUserID, senderBankCode, receiverUserID, receiverBankCode, amount, transaction_type } = req.body;

    try {
        // Validate banks
        const senderBank = await Bank.findOne({ bankCode: senderBankCode });
        const receiverBank = await Bank.findOne({ bankCode: receiverBankCode });
        if (!senderBank || !receiverBank) return res.status(400).json({ message: 'Sender or Receiver bank not found' });

        // Validate users
        const sender = await User.findOne({ userID: senderUserID, bankID: senderBank._id });
        const receiver = await User.findOne({ userID: receiverUserID, bankID: receiverBank._id });
        if (!sender || !receiver) return res.status(400).json({ message: 'Sender or Receiver user not found in respective banks' });

        // Choose transaction model for sender bank
        const TransactionModel = getTransactionModel(senderBankCode);

        const transaction = await TransactionModel.create({
            senderUserID: sender._id,
            receiverUserID: receiver._id,
            amount,
            transaction_type,
            sender_account: sender.userID,
            receiver_account: receiver.userID,
            ip_address: req.ip,
            device_hash: 'dummy-device-hash', // you can replace with actual device hash
        });

        res.json({ message: 'Transaction successful', transaction });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

module.exports = router;
