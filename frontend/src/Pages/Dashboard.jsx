import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { ArrowUpCircleIcon, ArrowDownCircleIcon, ExclamationTriangleIcon } from '@heroicons/react/24/solid';

const Dashboard = ({ user }) => {
    const [transactions, setTransactions] = useState([]);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState('');

    useEffect(() => {
        if (!user) return;

        const fetchUserTransactions = async () => {
            setIsLoading(true);
            setError('');
            try {
                const response = await axios.get(`/api/transactions/${user.userID}/${user.bankCode}`);
                setTransactions(response.data);
            } catch (err) {
                setError('Failed to load transaction history. Please try again later.');
                console.error(err);
            } finally {
                setIsLoading(false);
            }
        };

        fetchUserTransactions();
    }, [user]); // Re-fetch if the user object changes

    const renderTransactionList = () => {
        if (isLoading) {
            return <p className="text-gray-500">Loading history...</p>;
        }

        if (error) {
            return <p className="p-4 text-red-700 bg-red-100 rounded-md">{error}</p>;
        }

        if (transactions.length === 0) {
            return <p className="text-gray-500">No transactions found.</p>;
        }

        return (
            <ul className="space-y-4">
                {transactions.map((tx) => {
                    // Determine if the transaction is outgoing (sent by the current user)
                    const isOutgoing = tx.senderUserID._id === user._id;

                    return (
                        <li key={tx._id} className="bg-white p-4 rounded-lg shadow-md border-l-4 transition-transform hover:scale-[1.02] duration-200"
                            style={{ borderLeftColor: isOutgoing ? '#ef4444' : '#22c55e' }}>
                            <div className="flex justify-between items-center">
                                <div className="flex items-center space-x-4">
                                    {isOutgoing ? (
                                        <ArrowUpCircleIcon className="h-8 w-8 text-red-500" />
                                    ) : (
                                        <ArrowDownCircleIcon className="h-8 w-8 text-green-500" />
                                    )}
                                    <div>
                                        <p className="font-semibold text-gray-800">
                                            {isOutgoing ? `Sent to ${tx.receiver_account}` : `Received from ${tx.sender_account}`}
                                        </p>
                                        <p className="text-sm text-gray-500">{new Date(tx.timestamp).toLocaleString()}</p>
                                    </div>
                                </div>
                                <div className="text-right">
                                    <p className={`text-lg font-bold ${isOutgoing ? 'text-red-600' : 'text-green-600'}`}>
                                        {isOutgoing ? '-' : '+'} ${Number(tx.amount).toFixed(2)}
                                    </p>
                                    {tx.isFraudulent && (
                                        <div className="flex items-center justify-end mt-1 text-xs text-red-700 font-semibold">
                                            <ExclamationTriangleIcon className="h-4 w-4 mr-1" />
                                            FLAGGED
                                        </div>
                                    )}
                                </div>
                            </div>
                        </li>
                    );
                })}
            </ul>
        );
    };

    return (
        <div className="max-w-4xl mx-auto p-4 sm:p-6 lg:p-8 font-sans">
            <header className="pb-4 mb-8">
                <h1 className="text-3xl font-bold text-gray-800">Transaction History</h1>
                <p className="text-gray-500">A record of your recent activity.</p>
            </header>
            
            <main>
                {renderTransactionList()}
            </main>
        </div>
    );
};

export default Dashboard;