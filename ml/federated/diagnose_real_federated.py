#!/usr/bin/env python3
"""
Real Federated Learning Diagnostic

Tests the actual federated learning setup to find the 4KB model issue.
"""

import sys
import os
import pickle
import lightgbm as lgb
import numpy as np
import pandas as pd

# Add paths
sys.path.append('./models')
sys.path.append('./config')

def check_model_parameter_mismatch():
    """Check if there's a model parameter mismatch causing issues."""
    print("🔍 Checking Model Parameter Mismatch")
    print("=" * 60)
    
    # Load original model
    original_path = "./trained_models/lightgbm_model.pkl"
    if not os.path.exists(original_path):
        print(f"❌ Original model not found: {original_path}")
        return
    
    with open(original_path, 'rb') as f:
        original_model = pickle.load(f)
    
    print("📊 Original Model (from your training):")
    if hasattr(original_model, 'get_params'):
        original_params = original_model.get_params()
        print(f"  n_estimators: {original_params.get('n_estimators', 'N/A')}")
        print(f"  learning_rate: {original_params.get('learning_rate', 'N/A')}")
        print(f"  max_depth: {original_params.get('max_depth', 'N/A')}")
        print(f"  num_leaves: {original_params.get('num_leaves', 'N/A')}")
    
    if hasattr(original_model, 'booster_'):
        print(f"  Number of trees: {original_model.booster_.num_trees()}")
        
        # Get model dump to test loading
        model_dump = original_model.booster_.model_to_string()
        print(f"  Model dump length: {len(model_dump)} characters")
    
    # Check federated client parameters
    print("\n📊 Federated Client Model (from lightgbm_bank_client.py):")
    client_params = {
        'n_estimators': 50,    # Different!
        'learning_rate': 0.1,  # Different!
        'max_depth': 6,        # Different!
        'num_leaves': 31,      # Same
        'random_state': 42     # Same
    }
    
    for key, value in client_params.items():
        if hasattr(original_model, 'get_params'):
            original_value = original_params.get(key, 'N/A')
            if original_value != value:
                print(f"  {key}: {value} ❌ (original: {original_value})")
            else:
                print(f"  {key}: {value} ✅")
        else:
            print(f"  {key}: {value}")
    
    # Test if model dump from 200-tree model can load into 50-tree model
    print("\n🧪 Testing Model Dump Compatibility:")
    if hasattr(original_model, 'booster_'):
        try:
            # Create client-style model
            from models.lightgbm_model import FederatedLightGBM
            fed_model = FederatedLightGBM(
                n_estimators=50,
                learning_rate=0.1, 
                max_depth=6,
                num_leaves=31,
                random_state=42
            )
            
            # Try to load original model dump
            model_dump = original_model.booster_.model_to_string()
            
            # Simulate what happens in set_model_params
            test_params = {
                'model_dump': model_dump,
                'params': client_params,
                'total_samples': 10000,
                'model_type': 'lightgbm'
            }
            
            print("  Attempting to load 200-tree model into 50-tree client...")
            fed_model.set_model_params(test_params)
            
            if fed_model.is_fitted and hasattr(fed_model.model, '_Booster'):
                actual_trees = fed_model.model._Booster.num_trees()
                print(f"  ✅ SUCCESS: Loaded model with {actual_trees} trees")
            else:
                print("  ❌ FAILED: Model not loaded properly")
                print("  🚨 This is likely the cause of 4KB models!")
                
        except Exception as e:
            print(f"  ❌ ERROR loading model dump: {e}")
            print("  🚨 This confirms the parameter mismatch issue!")

def test_model_parameter_fixing():
    """Test fixing the model parameters in the client."""
    print("\n🔧 Testing Model Parameter Fix")
    print("=" * 60)
    
    # Load original model parameters
    original_path = "./trained_models/lightgbm_model.pkl"
    with open(original_path, 'rb') as f:
        original_model = pickle.load(f)
    
    if not hasattr(original_model, 'get_params'):
        print("❌ Cannot get original model parameters")
        return
    
    original_params = original_model.get_params()
    
    print("🔄 Creating FederatedLightGBM with ORIGINAL parameters:")
    print(f"  n_estimators: {original_params.get('n_estimators', 100)}")
    print(f"  learning_rate: {original_params.get('learning_rate', 0.1)}")
    print(f"  max_depth: {original_params.get('max_depth', 6)}")
    
    try:
        from models.lightgbm_model import FederatedLightGBM
        
        # Create model with CORRECT parameters
        fed_model = FederatedLightGBM(
            n_estimators=original_params.get('n_estimators', 200),
            learning_rate=original_params.get('learning_rate', 0.05),
            max_depth=original_params.get('max_depth', 10),
            num_leaves=original_params.get('num_leaves', 31),
            random_state=original_params.get('random_state', 42)
        )
        
        # Test loading model dump
        model_dump = original_model.booster_.model_to_string()
        
        test_params = {
            'model_dump': model_dump,
            'params': original_params,
            'total_samples': 10000,
            'model_type': 'lightgbm'
        }
        
        print("\n🔄 Loading model dump into correctly parameterized model...")
        fed_model.set_model_params(test_params)
        
        if fed_model.is_fitted and hasattr(fed_model.model, '_Booster'):
            actual_trees = fed_model.model._Booster.num_trees()
            print(f"  ✅ SUCCESS: Loaded model with {actual_trees} trees")
            
            # Test serialization size
            saved_params = fed_model.get_model_params()
            if saved_params:
                dump_size = len(saved_params.get('model_dump', ''))
                print(f"  ✅ Model dump size: {dump_size} characters")
                
                if dump_size > 50000:  # Should be large for 200 trees
                    print("  ✅ Model dump size looks correct (>50K chars)")
                    return True
                else:
                    print("  ⚠️  Model dump size seems small")
                    
        else:
            print("  ❌ Model not loaded properly")
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
        
    return False

def main():
    """Main diagnostic function."""
    print("🚀 Real Federated Learning Diagnostic")
    print("🎯 Finding the cause of 4KB models")
    print("=" * 80)
    
    # Check parameter mismatch
    check_model_parameter_mismatch()
    
    # Test the fix
    success = test_model_parameter_fixing()
    
    print("\n" + "=" * 80)
    if success:
        print("🎉 DIAGNOSIS COMPLETE - SOLUTION FOUND!")
        print("")
        print("💡 THE PROBLEM:")
        print("   Client model parameters don't match original model")
        print("   • Original: 200 trees, lr=0.05, depth=10")
        print("   • Client:   50 trees,  lr=0.1,  depth=6")
        print("")
        print("🔧 THE SOLUTION:")
        print("   Update lightgbm_bank_client.py line 49-55 to use:")
        print("   n_estimators=200, learning_rate=0.05, max_depth=10")
    else:
        print("❌ Issue not resolved. Further investigation needed.")

if __name__ == "__main__":
    main()
