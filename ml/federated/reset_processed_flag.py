#!/usr/bin/env python3
"""
Reset the processed_for_fl flag in MongoDB to allow reprocessing of data.
"""

import sys
import os
import pymongo

# Add project directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'config'))

from config.federated_config import get_bank_config

def reset_processed_flags():
    """Reset processed_for_fl flags in MongoDB."""
    
    print("🔄 Resetting processed_for_fl flags")
    print("=" * 40)
    
    try:
        # Get SBI configuration
        bank_config = get_bank_config("SBI")
        mongo_config = bank_config["mongo_config"]
        
        print(f"🏦 Bank: {bank_config['bank_id']}")
        print(f"🗄️ Database: {mongo_config['database']}")
        print(f"📁 Collection: {mongo_config['collection_template']}")
        
        # Connect to MongoDB
        client = pymongo.MongoClient(mongo_config["connection_string"])
        db = client[mongo_config["database"]]
        collection = db[mongo_config["collection_template"]]
        
        # Count total documents
        total_docs = collection.count_documents({})
        print(f"📊 Total documents: {total_docs}")
        
        # Count processed documents
        processed_docs = collection.count_documents({"processed_for_fl": True})
        print(f"⚙️ Previously processed: {processed_docs}")
        
        # Reset all processed_for_fl flags
        result = collection.update_many(
            {"processed_for_fl": True},
            {"$set": {"processed_for_fl": False}}
        )
        
        print(f"✅ Reset {result.modified_count} documents")
        
        # Verify reset
        remaining_processed = collection.count_documents({"processed_for_fl": True})
        print(f"🔍 Documents still marked as processed: {remaining_processed}")
        
        # Count available for FL
        available_docs = collection.count_documents({
            "bank_id": "SBI",
            "processed_for_fl": {"$ne": True}
        })
        print(f"🎯 Documents available for FL: {available_docs}")
        
        client.close()
        return True
        
    except Exception as e:
        print(f"❌ Reset failed: {e}")
        return False

if __name__ == "__main__":
    success = reset_processed_flags()
    
    if success:
        print("\n🎉 All documents are now available for federated learning!")
        print("   You can restart the client to process all your data.")
    else:
        print("\n⚠️ Reset failed. Check MongoDB connection.")
