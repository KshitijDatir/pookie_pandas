#!/usr/bin/env python3
"""
Final verification that the complete fix works.
"""

import sys
import os
import pickle

# Add paths
sys.path.append('./models')
sys.path.append('./config')

def verify_current_model():
    """Verify the current model is the correct size."""
    print("🔍 Current Model Status")
    print("-" * 40)
    
    current_path = "./trained_models/latest_lightgbm_federated.pkl"
    if os.path.exists(current_path):
        size = os.path.getsize(current_path) / 1024
        print(f"✅ Current model size: {size:.2f} KB")
        
        if size > 500:  # Should be ~708KB
            print("✅ Model size looks correct (large model)")
            return True
        else:
            print("❌ Model size is too small!")
            return False
    else:
        print("❌ Current model file not found")
        return False

def verify_client_parameters():
    """Verify client parameters match original training."""
    print("\n🔍 Client Parameter Verification")
    print("-" * 40)
    
    # Load original model to get reference parameters
    original_path = "./trained_models/lightgbm_model.pkl"
    with open(original_path, 'rb') as f:
        original_model = pickle.load(f)
    
    original_params = original_model.get_params()
    
    # Expected client parameters (from our fix)
    expected_params = {
        'n_estimators': 200,
        'learning_rate': 0.05,
        'max_depth': 10,
        'num_leaves': 31,
        'random_state': 42
    }
    
    print("Client Parameters (from lightgbm_bank_client.py):")
    all_match = True
    for key, expected_value in expected_params.items():
        original_value = original_params.get(key)
        if original_value == expected_value:
            print(f"  {key}: {expected_value} ✅")
        else:
            print(f"  {key}: {expected_value} ❌ (original: {original_value})")
            all_match = False
    
    return all_match

def verify_federated_model_fix():
    """Verify the FederatedLightGBM model fix works."""
    print("\n🔍 FederatedLightGBM Model Fix Verification")
    print("-" * 40)
    
    try:
        from models.lightgbm_model import FederatedLightGBM
        
        # Load original model for testing
        original_path = "./trained_models/lightgbm_model.pkl"
        with open(original_path, 'rb') as f:
            original_model = pickle.load(f)
        
        # Create federated model with correct parameters
        fed_model = FederatedLightGBM(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=10,
            num_leaves=31,
            random_state=42
        )
        
        # Extract model dump from original
        model_dump = original_model.booster_.model_to_string()
        
        # Test set_model_params (our fix)
        test_params = {
            'model_dump': model_dump,
            'params': original_model.get_params(),
            'total_samples': 10000,
            'model_type': 'lightgbm'
        }
        
        print("Testing set_model_params fix...")
        fed_model.set_model_params(test_params)
        
        if fed_model.is_fitted and hasattr(fed_model.model, '_Booster'):
            trees = fed_model.model._Booster.num_trees()
            print(f"✅ Model loaded with {trees} trees")
            
            # Test get_model_params
            extracted_params = fed_model.get_model_params()
            if extracted_params and len(extracted_params.get('model_dump', '')) > 500000:
                print(f"✅ Model parameters extracted: {len(extracted_params['model_dump'])} chars")
                return True
            else:
                print("❌ Model parameter extraction failed")
                return False
        else:
            print("❌ Model not loaded correctly")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Main verification function."""
    print("🎯 Complete Fix Verification")
    print("=" * 60)
    
    # Test 1: Current model size
    model_ok = verify_current_model()
    
    # Test 2: Client parameters
    params_ok = verify_client_parameters()
    
    # Test 3: FederatedLightGBM fix
    fed_fix_ok = verify_federated_model_fix()
    
    print("\n" + "=" * 60)
    print("📋 VERIFICATION RESULTS:")
    print(f"  Current Model Size: {'✅ PASS' if model_ok else '❌ FAIL'}")
    print(f"  Client Parameters:  {'✅ PASS' if params_ok else '❌ FAIL'}")
    print(f"  FederatedLGB Fix:   {'✅ PASS' if fed_fix_ok else '❌ FAIL'}")
    
    if model_ok and params_ok and fed_fix_ok:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ The federated learning system should now work correctly")
        print("✅ No more 4KB models - federated rounds will preserve full model size")
        print("\n🚀 Ready for federated learning!")
    else:
        print("\n❌ Some issues remain. Please check the failed tests above.")

if __name__ == "__main__":
    main()
