import { useState, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import axios from "axios";

export default function User() {
  const { userID } = useParams();
  const navigate = useNavigate();

  const [transactions, setTransactions] = useState([]);
  const [bankCode, setBankCode] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    const fetchTransactions = async () => {
      setLoading(true);
      setError("");
      try {
        const response = await axios.get(`http://localhost:5000/api/getTransactions/${userID}`);
        setTransactions(response.data.transactions);
        setBankCode(response.data.bankCode);
      } catch (err) {
        setError(err.response?.data?.message || "Failed to fetch transactions");
      } finally {
        setLoading(false);
      }
    };

    fetchTransactions();
  }, [userID]);

  const handleDoTransaction = () => {
    navigate(`/dotransaction/${userID}`);
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-start bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 p-6">
      <div className="bg-white p-8 rounded-2xl shadow-2xl w-full max-w-3xl border border-slate-200">
        {/* Header */}
        <div className="text-center mb-6">
          <h2 className="text-3xl font-bold text-slate-800 mb-2">Transactions for {userID}</h2>
          {loading && <p className="text-slate-500">Fetching transactions...</p>}
        </div>

        {/* Error Message */}
        {error && (
          <div className="mb-4 p-4 rounded-xl bg-red-50 border border-red-200 text-red-800 text-center">
            {error}
          </div>
        )}

        {/* Transactions Table */}
        {!loading && transactions.length > 0 && (
          <div>
            <p className="mb-2 text-slate-700 font-semibold">Bank Code: {bankCode}</p>
            <table className="w-full table-auto border-collapse border border-slate-300">
              <thead className="bg-slate-100">
                <tr>
                  <th className="border border-slate-300 px-4 py-2">Date</th>
                  <th className="border border-slate-300 px-4 py-2">Type</th>
                  <th className="border border-slate-300 px-4 py-2">Amount</th>
                  <th className="border border-slate-300 px-4 py-2">Status</th>
                </tr>
              </thead>
              <tbody>
                {transactions.map((tx, index) => (
                  <tr key={index} className="hover:bg-slate-50">
                    <td className="border border-slate-300 px-4 py-2">{tx.date || "-"}</td>
                    <td className="border border-slate-300 px-4 py-2">{tx.type || "-"}</td>
                    <td className="border border-slate-300 px-4 py-2">{tx.amount || "-"}</td>
                    <td className="border border-slate-300 px-4 py-2">{tx.status || "-"}</td>
                  </tr>
                ))}
              </tbody>
            </table>

            {/* Do Transaction Button */}
            <div className="mt-6 text-center">
              <button
                onClick={handleDoTransaction}
                className="px-6 py-3 rounded-xl font-semibold text-white bg-green-600 hover:bg-green-700 transition-colors duration-200 shadow-lg"
              >
                Do Transaction
              </button>
            </div>
          </div>
        )}

        {/* No transactions */}
        {!loading && transactions.length === 0 && !error && (
          <p className="text-center text-slate-500 mt-6">No transactions found.</p>
        )}
      </div>
    </div>
  );
}