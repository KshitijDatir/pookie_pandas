// src/components/Navbar.jsx

import React from 'react';
import { NavLink, useLocation } from 'react-router-dom';

const Navbar = () => {
    const location = useLocation(); // Get current route
    console.log(location.pathname);
    // Don't show Navbar on the login page
    if (location.pathname === "/") return null;

    const activeLinkStyle = {
        backgroundColor: '#3b82f6', // Tailwind's blue-600
        color: 'white',
    };

    return (
        <header className="bg-white shadow-md sticky top-0 z-50">
            <nav className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="flex justify-between items-center h-16">
                    {/* App Logo/Name */}
                    <div className="flex-shrink-0 font-bold text-xl text-gray-800">
                        PookieBank
                    </div>

                    {/* Navigation Links */}
                    { (
                        <div className="hidden md:flex items-center space-x-4">
                            <NavLink
                                to="/transaction"
                                style={({ isActive }) => (isActive ? activeLinkStyle : undefined)}
                                className="px-3 py-2 rounded-md text-sm font-medium text-gray-700 hover:bg-gray-200"
                            >
                                Make Transaction
                            </NavLink>
                            <NavLink
                                to="/history"
                                style={({ isActive }) => (isActive ? activeLinkStyle : undefined)}
                                className="px-3 py-2 rounded-md text-sm font-medium text-gray-700 hover:bg-gray-200"
                            >
                                History
                            </NavLink>
                        </div>
                    )}

                    {/* User Info and Logout */}
                    { (
                        <div className="flex items-center space-x-4">
                          
                            <button
                                onClick={() => {
                                    localStorage.removeItem('token');
                                    window.location.href = '/';
                                }}
                                className="px-3 py-2 bg-red-600 text-white text-sm font-medium rounded-md hover:bg-red-700"
                            >
                                Logout
                            </button>
                        </div>
                    )}
                </div>
            </nav>
        </header>
    );
};

export default Navbar;
