const express = require('express');
const router = express.Router();
const User = require('../schemas/user.js');
const Bank = require('../schemas/bank.js');
const { Bank1Transaction, Bank2Transaction, Bank3Transaction } = require('../schemas/transaction.js');
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
