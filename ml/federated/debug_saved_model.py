#!/usr/bin/env python3
"""
Debug script to examine what's actually in the saved federated model files.
"""

import pickle
import json
import os

def examine_model_file(filepath):
    """Examine a model file and report its contents."""
    print(f"\n🔍 Examining: {filepath}")
    
    if not os.path.exists(filepath):
        print(f"❌ File does not exist")
        return
    
    file_size = os.path.getsize(filepath)
    print(f"📊 File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
    
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        print(f"🔧 Data type: {type(data)}")
        
        if isinstance(data, dict):
            print(f"📋 Dictionary keys: {list(data.keys())}")
            
            # Check key components
            if 'model_dump' in data:
                model_dump = data['model_dump']
                print(f"📊 model_dump length: {len(model_dump):,} chars")
                print(f"📊 model_dump preview: {str(model_dump)[:100]}...")
                
                # Check if it's a valid LightGBM model string
                if 'tree' in model_dump.lower():
                    print("✅ Contains tree information")
                else:
                    print("❌ No tree information found")
            
            if 'num_trees' in data:
                print(f"🌳 Number of trees: {data['num_trees']}")
            
            if 'total_samples' in data:
                print(f"📈 Total samples: {data['total_samples']:,}")
                
            if 'params' in data:
                params = data['params']
                print(f"⚙️ Parameters: {params}")
        
        else:
            print(f"🔧 Non-dict data: {str(data)[:200]}")
            
    except Exception as e:
        print(f"❌ Failed to load: {e}")

def main():
    print("🚀 Debugging Saved Federated Models")
    
    # Check both model files
    examine_model_file("trained_models/latest_lightgbm_federated.pkl")
    examine_model_file("trained_models/lightgbm_model.pkl")
    
    # Also check if there are version files
    versions_dir = "trained_models/versions"
    if os.path.exists(versions_dir):
        print(f"\n📁 Checking versions directory...")
        for file in os.listdir(versions_dir):
            if file.endswith('.pkl'):
                filepath = os.path.join(versions_dir, file)
                examine_model_file(filepath)

if __name__ == "__main__":
    main()
