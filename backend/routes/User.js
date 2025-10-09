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
        console.log(bankCode);
       
        const user = await User.findOne({ userID, bankCode });
        if (!user) return res.status(400).json({ message: 'User not found in this bank' });

        res.json({ message: 'Login successful', user });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});
router.get('/getTransactions/:userID', async (req, res) => {
  const { userID } = req.params;

  try {
    // Find the user first
    const user = await User.findOne({ userID });
    if (!user) return res.status(400).json({ message: 'User not found' });

    let transactions;

    // Fetch transactions depending on bankCode
    switch (user.bankCode) {
      case 'bank1':
        transactions = await Bank1Transaction.find({ userID });
        break;
      case 'bank2':
        transactions = await Bank2Transaction.find({ userID });
        break;
      case 'bank3':
        transactions = await Bank3Transaction.find({ userID });
        break;
      default:
        return res.status(400).json({ message: 'Invalid bankCode' });
    }

    res.json({ userID, bankCode: user.bankCode, transactions });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// In routes/user.js
// routes/user.js
router.post('/register', async (req, res) => {
  try {
    const { userID, name, email, phone, bankCode, balance } = req.body;

    // Find the bank by bankCode
    const bank = await Bank.findOne({ bankCode });
    if (!bank) return res.status(400).json({ message: 'Bank not found' });

    // Create the user using bank._id
    const user = await User.create({
      userID,
      name,
      email,
      phone,
      bankCode,
      balance: balance || 0
    });

    // Add user to bank's users array
    bank.users.push(user._id);
    await bank.save();

    res.status(201).json({ message: 'User registered successfully', user });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});


module.exports = router;
