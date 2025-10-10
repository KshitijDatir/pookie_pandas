import React, { createContext, useContext, useState, useEffect } from 'react';
import { CheckCircleIcon, ExclamationCircleIcon, InformationCircleIcon } from '@heroicons/react/24/solid';

// Toast Context
const ToastContext = createContext();

// Toast Provider Component
export const ToastProvider = ({ children }) => {
    const [toasts, setToasts] = useState([]);

    const addToast = (message, type = 'info', duration = 4000) => {
        const id = Date.now() + Math.random();
        const newToast = { id, message, type, duration };
        
        setToasts(prev => [...prev, newToast]);

        // Auto remove toast after duration
        setTimeout(() => {
            removeToast(id);
        }, duration);

        return id;
    };

    const removeToast = (id) => {
        setToasts(prev => prev.filter(toast => toast.id !== id));
    };

    return (
        <ToastContext.Provider value={{ addToast, removeToast }}>
            {children}
            <ToastContainer toasts={toasts} onRemove={removeToast} />
        </ToastContext.Provider>
    );
};

// Toast Container Component
const ToastContainer = ({ toasts, onRemove }) => {
    if (toasts.length === 0) return null;

    return (
        <div className="fixed top-4 right-4 z-50 space-y-2">
            {toasts.map(toast => (
                <Toast key={toast.id} toast={toast} onRemove={onRemove} />
            ))}
        </div>
    );
};

// Individual Toast Component
const Toast = ({ toast, onRemove }) => {
    const [isVisible, setIsVisible] = useState(false);
    const [isLeaving, setIsLeaving] = useState(false);

    useEffect(() => {
        // Trigger entrance animation
        const timer = setTimeout(() => setIsVisible(true), 10);
        return () => clearTimeout(timer);
    }, []);

    const handleClose = () => {
        setIsLeaving(true);
        setTimeout(() => {
            onRemove(toast.id);
        }, 300);
    };

    const getToastStyles = () => {
        const baseStyles = "flex items-center p-4 rounded-lg shadow-lg border transition-all duration-300 transform";
        const visibilityStyles = isVisible && !isLeaving 
            ? "translate-x-0 opacity-100" 
            : "translate-x-full opacity-0";

        switch (toast.type) {
            case 'success':
                return `${baseStyles} bg-green-50 border-green-200 text-green-800 ${visibilityStyles}`;
            case 'error':
                return `${baseStyles} bg-red-50 border-red-200 text-red-800 ${visibilityStyles}`;
            case 'info':
            default:
                return `${baseStyles} bg-blue-50 border-blue-200 text-blue-800 ${visibilityStyles}`;
        }
    };

    const getIcon = () => {
        const iconClass = "h-5 w-5 mr-3 flex-shrink-0";
        switch (toast.type) {
            case 'success':
                return <CheckCircleIcon className={`${iconClass} text-green-600`} />;
            case 'error':
                return <ExclamationCircleIcon className={`${iconClass} text-red-600`} />;
            case 'info':
            default:
                return <InformationCircleIcon className={`${iconClass} text-blue-600`} />;
        }
    };

    return (
        <div className={getToastStyles()}>
            {getIcon()}
            <p className="flex-1 text-sm font-medium">{toast.message}</p>
            <button
                onClick={handleClose}
                className="ml-3 text-gray-400 hover:text-gray-600 transition-colors"
            >
                <span className="sr-only">Close</span>
                <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
            </button>
        </div>
    );
};

// Custom Hook to use Toast
export const useToast = () => {
    const context = useContext(ToastContext);
    if (!context) {
        throw new Error('useToast must be used within a ToastProvider');
    }
    return context;
};

export default Toast;
