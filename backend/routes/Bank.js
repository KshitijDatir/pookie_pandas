const express = require('express');
const router = express.Router();
const User = require('../schemas/user.js');
const Bank = require('../schemas/bank.js');
const { Bank1Transaction, Bank2Transaction, Bank3Transaction } = require('../schemas/transaction.js');

// In routes/bank.js
router.post('/banks', async (req, res) => {
  try {
    const bank = await Bank.create(req.body);
    res.status(201).json({ message: "Bank created", bank });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});
module.exports = router;