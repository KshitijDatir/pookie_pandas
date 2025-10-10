// Test script for TXN collections
const http = require('http');

const testData = {
    sender_account: "TXNTEST123456",
    receiver_account: "TXNTEST654321", 
    amount: 2500.50,
    transaction_type: "payment",
    merchant_category: "grocery",
    bank_id: "SBI"
};

const postData = JSON.stringify(testData);

const options = {
    hostname: 'localhost',
    port: 5000,
    path: '/api/simulate-transaction',
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(postData)
    }
};

console.log('🧪 Testing TXN Collection Storage...');
console.log('📤 Sending transaction data:', testData);

const req = http.request(options, (res) => {
    console.log(`✅ Status: ${res.statusCode}`);
    
    let data = '';
    res.on('data', (chunk) => {
        data += chunk;
    });
    
    res.on('end', () => {
        const response = JSON.parse(data);
        console.log('📥 Response:');
        console.log('  Message:', response.message);
        console.log('  Transaction ID:', response.transactionId);
        console.log('  Bank:', response.bank);
        console.log('  Collection:', response.collection);
        console.log('  Amount:', response.transaction.amount);
        
        if (response.collection === 'sbi_txns') {
            console.log('🎯 SUCCESS: Transaction saved to TXN collection!');
        } else {
            console.log('❌ ERROR: Transaction saved to wrong collection:', response.collection);
        }
    });
});

req.on('error', (e) => {
    console.error(`❌ Problem with request: ${e.message}`);
});

req.write(postData);
req.end();
