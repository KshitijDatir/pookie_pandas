const express = require('express');
const router = express.Router();
const User = require('../schemas/user.js');
const Bank = require('../schemas/bank.js');
const { Bank1Transaction, Bank2Transaction, Bank3Transaction } = require('../schemas/transaction.js');

function getTransactionModel(bankCode) {
    switch(bankCode) {
        case 'BANK1': return Bank1Transaction;
        case 'BANK2': return Bank2Transaction;
        case 'BANK3': return Bank3Transaction;
        default: throw new Error('Invalid bank code');
    }
}
