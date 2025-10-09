import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useParams } from 'react-router-dom';


const User = ({ user, onLogout }) => {
    const { userID } = useParams();
    // State for the transaction form inputs
    const [receiverUserID, setReceiverUserID] = useState('');
    const [receiverBankCode, setReceiverBankCode] = useState('');
    const [amount, setAmount] = useState('');
    const [transactionType, setTransactionType] = useState('TRANSFER');
    
    // --- NEW STATE for the fraud toggle ---
    const [isFraudulent, setIsFraudulent] = useState(false);

    // State for past transactions
    const [transactions, setTransactions] = useState([]);

    // State for UI feedback
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState('');
    const [successMessage, setSuccessMessage] = useState('');

    const fetchTransactions = async () => { /* ... same as before ... */ };

    useEffect(() => {
        if (user) fetchTransactions();
    }, [user]);

    // --- UPDATED handleMakeTransaction function ---
    const handleMakeTransaction = async (e) => {
        e.preventDefault();
        setError('');
        setSuccessMessage('');
        setIsLoading(true);

        const transactionData = {
            senderUserID: user.userID,
            senderBankCode: user.bankCode,
            receiverUserID,
            receiverBankCode,
            amount: Number(amount),
            transaction_type: transactionType,
            // --- NEW field being sent to the backend ---
            isFraudulent: isFraudulent,
        };

        try {
            // The API call remains the same, but the payload is now richer
            const response = await axios.post('/api/transaction', transactionData);
            setSuccessMessage(`Transaction successful! ID: ${response.data.transaction._id}`);
            
            // Clear the form fields and reset the toggle
            setReceiverUserID('');
            setReceiverBankCode('');
            setAmount('');
            setIsFraudulent(false); // Reset toggle to "Normal"
            
            setTimeout(() => fetchTransactions(), 1000);
        } catch (err) {
            const errorMessage = err.response?.data?.message || 'Transaction failed. Please try again.';
            setError(errorMessage);
        } finally {
            setIsLoading(false);
        }
    };

    if (!user) {
        return <div className="text-center p-10">Loading user data...</div>;
    }

    return (
        <div className="max-w-4xl mx-auto p-4 sm:p-6 lg:p-8 font-sans">
            <header className="flex justify-between items-center pb-4 mb-8 border-b border-gray-300">
                <h1 className="text-2xl sm:text-3xl font-bold text-gray-800">Welcome, {user.name}</h1>
                <button
                    onClick={onLogout}
                    className="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 transition-colors duration-200"
                >
                    Logout
                </button>
            </header>

            <main className="space-y-8">
                {/* User Details Section (unchanged) */}
                <section className="bg-white p-6 rounded-lg shadow-md">
                    {/* ... same as before ... */}
                </section>

                {/* Make Transaction Section */}
                <section className="bg-white p-6 rounded-lg shadow-md">
                    <h2 className="text-xl font-semibold text-gray-700 mb-4">Make a Transaction</h2>
                    <form onSubmit={handleMakeTransaction} className="space-y-6">
                        {/* Sender Info (unchanged) */}
                        <div className="p-4 bg-gray-100 rounded-md border border-gray-200">
                           {/* ... same as before ... */}
                        </div>

                        {/* Receiver Info Inputs (unchanged) */}
                        <div className="space-y-4">
                           {/* ... all input fields for receiver and amount are the same ... */}
                        </div>

                        {/* --- NEW FRAUD TOGGLE BUTTON --- */}
                        <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg border">
                            <span className="text-sm font-medium text-gray-800">Transaction Mode</span>
                            <button
                                type="button" // Important: prevents form submission on click
                                onClick={() => setIsFraudulent(!isFraudulent)}
                                className={`px-4 py-2 text-sm font-semibold text-white rounded-md transition-all duration-300 transform active:scale-95 ${
                                    isFraudulent
                                        ? 'bg-red-500 hover:bg-red-600 shadow-md'
                                        : 'bg-green-500 hover:bg-green-600'
                                }`}
                            >
                                {isFraudulent ? 'Fraudulent Transaction' : 'Normal Transaction'}
                            </button>
                        </div>
                        
                        <button
                            type="submit"
                            disabled={isLoading}
                            className="w-full py-3 px-4 bg-blue-600 text-white font-semibold rounded-md hover:bg-blue-700 transition-colors duration-200 disabled:bg-gray-400 disabled:cursor-not-allowed"
                        >
                            {isLoading ? 'Processing...' : 'Send Money'}
                        </button>
                    </form>
                    {error && <p className="mt-4 p-3 rounded-md text-red-700 bg-red-100">{error}</p>}
                    {successMessage && <p className="mt-4 p-3 rounded-md text-green-700 bg-green-100">{successMessage}</p>}
                </section>

                {/* Past Transactions Section (unchanged) */}
                <section className="bg-white p-6 rounded-lg shadow-md">
                   {/* ... same as before ... */}
                </section>
            </main>
        </div>
    );
};

export default User;