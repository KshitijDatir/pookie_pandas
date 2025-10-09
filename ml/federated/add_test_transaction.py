#!/usr/bin/env python3
"""
Add Test Transaction to MongoDB

This script adds test transactions to MongoDB to help test the federated learning system.
"""

import os
import sys
import json
from datetime import datetime
from pymongo import MongoClient

# Add config path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'config'))

def add_test_transaction(bank_id: str = "SBI", num_transactions: int = 1, fraud: bool = False):
    """Add test transaction(s) to MongoDB."""
    
    try:
        from config.federated_config import get_bank_config
        
        bank_config = get_bank_config(bank_id)
        mongo_config = bank_config["mongo_config"]
        
        # Connect to MongoDB
        client = MongoClient(mongo_config["connection_string"])
        db = client[mongo_config["database"]]
        
        collection_name = mongo_config.get("collection_template", "sbi_qs")
        collection = db[collection_name]
        
        print(f"🔗 Connected to collection: {collection_name}")
        
        # Check current count
        current_count = collection.count_documents({"bank_id": bank_id, "processed_for_fl": {"$ne": True}})
        print(f"📊 Current unprocessed transactions for {bank_id}: {current_count}")
        
        # Generate test transactions
        transactions = []
        for i in range(num_transactions):
            transaction = {
                "bank_id": bank_id,
                "transaction_id": f"TEST_{bank_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}",
                "amount": 100.50 + (i * 10.25),
                "user_id": f"user_{100 + i}",
                "merchant_id": f"merchant_{200 + i}",
                "transaction_time": datetime.now(),
                "location_lat": 28.6139 + (i * 0.01),
                "location_long": 77.2090 + (i * 0.01),
                "account_age_days": 365 + i,
                "transaction_hour": (10 + i) % 24,
                "day_of_week": i % 7,
                "is_weekend": 0,
                "num_transactions_today": 1 + i,
                "avg_transaction_amount": 85.75 + (i * 5),
                "time_since_last_transaction": 3600 + (i * 100),
                "is_fraud": 1 if fraud else 0,
                "processed_for_fl": False,
                "created_at": datetime.now()
            }
            transactions.append(transaction)
        
        # Insert transactions
        result = collection.insert_many(transactions)
        print(f"✅ Added {len(result.inserted_ids)} test transactions")
        
        # Check new count
        new_count = collection.count_documents({"bank_id": bank_id, "processed_for_fl": {"$ne": True}})
        print(f"📊 New unprocessed transactions for {bank_id}: {new_count}")
        print(f"🎯 Federated learning threshold (5): {'✅ REACHED' if new_count >= 5 else '❌ NOT REACHED'}")
        
        # Show sample transaction
        if transactions:
            print(f"\n📋 Sample transaction:")
            print(f"   ID: {transactions[0]['transaction_id']}")
            print(f"   Amount: ${transactions[0]['amount']:.2f}")
            print(f"   Is Fraud: {transactions[0]['is_fraud']}")
        
        client.close()
        
    except Exception as e:
        print(f"❌ Error adding test transaction: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Add test transactions to MongoDB')
    parser.add_argument('--bank', default='SBI', help='Bank ID (default: SBI)')
    parser.add_argument('--count', type=int, default=1, help='Number of transactions (default: 1)')
    parser.add_argument('--fraud', action='store_true', help='Mark transactions as fraud')
    
    args = parser.parse_args()
    
    print(f"Adding {args.count} test transaction(s) to {args.bank}")
    if args.fraud:
        print("WARNING: Marking as FRAUD transactions")
    
    add_test_transaction(
        bank_id=args.bank,
        num_transactions=args.count,
        fraud=args.fraud
    )

if __name__ == "__main__":
    main()
