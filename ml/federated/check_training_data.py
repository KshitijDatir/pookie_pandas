#!/usr/bin/env python3
"""
Check Training Data Availability

This checks if clients have sufficient training data.
"""

import sys
import os
from pymongo import MongoClient

# Add paths
sys.path.append('./config')

def check_mongodb_data():
    """Check actual data in MongoDB collections."""
    print("🔍 Checking MongoDB Training Data")
    print("=" * 60)
    
    try:
        from config.federated_config import get_bank_config
        
        banks = ['SBI', 'HDFC', 'AXIS']
        
        for bank in banks:
            print(f"\n🏦 {bank} Bank Data:")
            print("-" * 30)
            
            try:
                bank_config = get_bank_config(bank)
                mongo_config = bank_config["mongo_config"]
                
                # Connect to MongoDB
                client = MongoClient(mongo_config["connection_string"])
                db = client[mongo_config["database"]]
                collection_name = mongo_config["collection_template"]
                collection = db[collection_name]
                
                # Check total documents
                total_docs = collection.count_documents({})
                print(f"  Total documents: {total_docs}")
                
                # Check documents for this bank
                bank_docs = collection.count_documents({"bank_id": bank})
                print(f"  Documents with bank_id '{bank}': {bank_docs}")
                
                # Check unprocessed documents (available for FL)
                available_docs = collection.count_documents({
                    "bank_id": bank,
                    "processed_for_fl": {"$ne": True}
                })
                print(f"  Available for FL (unprocessed): {available_docs}")
                
                # Check processed documents
                processed_docs = collection.count_documents({
                    "bank_id": bank,
                    "processed_for_fl": True
                })
                print(f"  Already processed for FL: {processed_docs}")
                
                # Show sample document structure
                sample_doc = collection.find_one({"bank_id": bank})
                if sample_doc:
                    print(f"  Sample document keys: {list(sample_doc.keys())}")
                    print(f"  Has 'is_fraud' field: {'is_fraud' in sample_doc}")
                else:
                    print(f"  ❌ No documents found for bank {bank}")
                
                client.close()
                
                # Analysis
                if available_docs < 5:
                    print(f"  🚨 INSUFFICIENT DATA! Only {available_docs} < 5 required")
                elif available_docs < 100:
                    print(f"  ⚠️  LIMITED DATA: {available_docs} samples (models will be small)")
                else:
                    print(f"  ✅ SUFFICIENT DATA: {available_docs} samples")
                    
            except Exception as e:
                print(f"  ❌ Error checking {bank}: {e}")
    
    except Exception as e:
        print(f"❌ Error: {e}")

def simulate_client_training():
    """Simulate what happens during client training with limited data."""
    print("\n🧪 Simulating Client Training with Limited Data")
    print("=" * 60)
    
    try:
        import numpy as np
        import pandas as pd
        from models.lightgbm_model import FederatedLightGBM
        
        print("Testing with different data sizes:")
        
        for num_samples in [5, 10, 50, 100, 500]:
            print(f"\n📊 Training with {num_samples} samples:")
            
            # Create sample data
            np.random.seed(42)
            X = np.random.randn(num_samples, 10)
            y = (X[:, 0] + X[:, 1] > 0).astype(int)
            
            # Train federated model
            fed_model = FederatedLightGBM(
                n_estimators=200,  # Same as original
                learning_rate=0.05,
                max_depth=10,
                num_leaves=31,
                random_state=42
            )
            
            fed_model.fit(X, y)
            
            if fed_model.is_fitted:
                trees = fed_model.model.booster_.num_trees()
                print(f"  Trees created: {trees}")
                
                # Get model dump size
                params = fed_model.get_model_params()
                if params:
                    dump_size = len(params.get('model_dump', ''))
                    print(f"  Model dump size: {dump_size} characters")
                    print(f"  Expected file size: ~{dump_size/1024*1.1:.1f} KB")
                    
                    if trees < 50:
                        print(f"  🚨 TOO FEW TREES! This will create small models")
                    else:
                        print(f"  ✅ Adequate trees for full-size model")
            else:
                print(f"  ❌ Training failed")
                
    except Exception as e:
        print(f"❌ Error in simulation: {e}")

def main():
    """Main function."""
    print("🎯 Training Data Analysis")
    print("=" * 80)
    
    # Check MongoDB data
    check_mongodb_data()
    
    # Simulate training with different data sizes
    simulate_client_training()
    
    print("\n" + "=" * 80)
    print("💡 ANALYSIS SUMMARY:")
    print("")
    print("If clients have < 100 samples:")
    print("  → LightGBM creates very few trees")
    print("  → Small models (few KB instead of hundreds of KB)")
    print("  → This explains the 3.4KB models!")
    print("")
    print("SOLUTION:")
    print("  1. Add more training data to MongoDB collections")
    print("  2. OR use the original large model for federated learning")
    print("  3. OR adjust LightGBM parameters for small datasets")

if __name__ == "__main__":
    main()
