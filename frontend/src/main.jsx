import React from "react";
import ReactDOM from "react-dom/client";
import Login from "./Components/Login.jsx";
import { BrowserRouter,Routes,Route } from 'react-router-dom';
import Dashboard from "./Pages/Dashboard.jsx";  
import User from "./Pages/user.jsx";
import TransactionSimulator from "./Pages/TransactionSimulator.jsx";
import "./index.css";
import Navbar from "./Components/navbar.jsx";

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <BrowserRouter>
       <Navbar/>
      <Routes>
          <Route path="/" element={<Login />} />
          <Route path="/user/:userID" element={<User />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/simulate" element={<TransactionSimulator />} />
      </Routes>
    
    </BrowserRouter>
  </React.StrictMode>
);
