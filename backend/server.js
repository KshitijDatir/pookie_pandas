// server.js
const express = require('express');
const cors = require('cors');
const app = express();
require('dotenv').config();
const PORT = process.env.PORT || 5000;
const mongoose = require('mongoose');
const userRoutes = require('./routes/User.js');
const bankRoutes = require('./routes/Bank.js');
const transactionRoutes = require('./routes/Transaction.js');
app.use(cors());
// CORS middleware
app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', 'http://localhost:5173');
  res.header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
  res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept, Authorization');
  
  // Handle preflight requests
  if (req.method === 'OPTIONS') {
    res.sendStatus(200);
  } else {
    next();
  }
});

// Middleware to parse JSON requests
app.use(express.json());
app.use('/api/user', userRoutes);
app.use('/api/bank', bankRoutes);
app.use('/api', transactionRoutes);
const mongoURI = process.env.MONGO_ATLAS_URL;
// Connect to MongoDB
mongoose.connect(mongoURI, {
  useNewUrlParser: true,
  useUnifiedTopology: true
})
.then(() => console.log('✅ MongoDB connected successfully!'))
.catch(err => console.error('❌ MongoDB connection error:', err.message));


app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
