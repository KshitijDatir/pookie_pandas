import { useState } from "react";
import axios from "axios";

export default function Login() {
  const [userID, setUserID] = useState("");
  const [bankCode, setBankCode] = useState("");
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setMessage("");
    setError("");
    setLoading(true);
    try {
      const response = await axios.post("http://localhost:5000/login", { userID, bankCode });
      setMessage(response.data.message);
      window.location.href = `/user/${userID}`; // Redirect to dashboard on successful login
    } catch (err) {
      setError(err.response?.data?.message || "Login failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-r from-blue-500 to-indigo-600 p-4">
      <div className="bg-white/10 backdrop-blur-md p-8 rounded-2xl shadow-2xl w-full max-w-md text-white border border-white/20">
        <h2 className="text-3xl font-bold mb-6 text-center tracking-wide">Bank Login</h2>
        <div className=""></div>
        <form onSubmit={handleSubmit} className="space-y-5">
          <div>
            <label className="block text-sm mb-2 font-medium">User ID</label>
            <input
              type="text"
              value={userID}
              onChange={(e) => setUserID(e.target.value)}
              placeholder="Enter your User ID"
              className="w-full px-4 py-2 rounded-lg bg-white/20 border border-white/30 text-white placeholder-gray-300 focus:outline-none focus:ring-2 focus:ring-white focus:bg-white/30 transition"
              required
            />
          </div>
          <div>
            <label className="block text-sm mb-2 font-medium">Bank Code</label>
            <input
              type="text"
              value={bankCode}
              onChange={(e) => setBankCode(e.target.value)}
              placeholder="Enter your Bank Code"
              className="w-full px-4 py-2 rounded-lg bg-white/20 border border-white/30 text-white placeholder-gray-300 focus:outline-none focus:ring-2 focus:ring-white focus:bg-white/30 transition"
              required
            />
          </div>
          <button
            type="submit"
            disabled={loading}
            className={`w-full py-2 rounded-lg font-semibold text-lg transition ${
              loading
                ? "bg-gray-400 cursor-not-allowed"
                : "bg-white text-blue-600 hover:bg-blue-100"
            }`}
          >
            {loading ? "Logging in..." : "Login"}
          </button>
        </form>
        {message && (
          <p className="mt-4 text-center text-green-300 font-medium animate-pulse">
            {message}
          </p>
        )}
        {error && (
          <p className="mt-4 text-center text-red-300 font-medium animate-pulse">
            {error}
          </p>
        )}
        <p className="mt-6 text-center text-sm text-white/70">
          Don’t have an account? <span className="underline cursor-pointer hover:text-white">Sign up</span>
        </p>
      </div>
    </div>
  );
}
