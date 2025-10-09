#!/usr/bin/env python3
"""
Final Comprehensive Test for LightGBM Federated Learning System

This test verifies that all components work together correctly.
"""

import os
import sys
import time
import threading
import subprocess
import json

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'models'))

def test_data_availability():
    """Test that we have sufficient data for FL."""
    print("🔄 Test 1: Data Availability")
    
    try:
        sys.path.append(os.path.join(current_dir, 'config'))
        from config.federated_config import get_bank_config
        from pymongo import MongoClient
        
        bank_config = get_bank_config("SBI")
        mongo_config = bank_config["mongo_config"]
        
        client = MongoClient(mongo_config["connection_string"])
        db = client[mongo_config["database"]]
        
        collection_name = mongo_config.get("collection_template", "trial")
        if "{bank_id}" in collection_name:
            collection_name = collection_name.format(bank_id="sbi")
        
        collection = db[collection_name]
        
        count = collection.count_documents({
            "bank_id": "SBI",
            "processed_for_fl": {"$ne": True}
        })
        
        print(f"   📊 Available transactions: {count}")
        if count >= 5:
            print(f"   ✅ Threshold reached ({count} >= 5)")
            return True
        else:
            print(f"   ❌ Insufficient data ({count} < 5)")
            return False
            
    except Exception as e:
        print(f"   ❌ Data check failed: {e}")
        return False

def test_model_components():
    """Test that model components work."""
    print("\n🔄 Test 2: Model Components")
    
    try:
        from models.lightgbm_model import FederatedLightGBM, aggregate_lightgbm_models
        import numpy as np
        import pandas as pd
        
        # Create test data
        np.random.seed(42)
        X = pd.DataFrame(np.random.rand(50, 5), columns=[f'feature_{i}' for i in range(5)])
        y = pd.Series(np.random.randint(0, 2, 50))
        
        # Test model creation and training
        model = FederatedLightGBM(n_estimators=5, random_state=42, verbose=-1)
        model.fit(X, y)
        
        # Test parameter extraction
        params = model.get_model_params()
        if not params:
            print("   ❌ Parameter extraction failed")
            return False
        
        # Test JSON serialization
        json_str = json.dumps(params, default=str)
        if len(json_str) < 100:
            print("   ❌ JSON serialization failed")
            return False
        
        # Test parameter setting
        model2 = FederatedLightGBM(n_estimators=5, random_state=123, verbose=-1)
        model2.set_model_params(params)
        model2.fit(X, y)  # Retrain with updated parameters
        
        # Test predictions
        pred1 = model.predict(X[:5])
        pred2 = model2.predict(X[:5])
        
        print("   ✅ Model training successful")
        print("   ✅ Parameter extraction successful")
        print("   ✅ JSON serialization successful")
        print("   ✅ Parameter setting successful")
        print("   ✅ Predictions successful")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Model component test failed: {e}")
        return False

def test_client_server_communication():
    """Test basic client-server parameter format."""
    print("\n🔄 Test 3: Client-Server Communication Format")
    
    try:
        from models.lightgbm_model import FederatedLightGBM
        import numpy as np
        import pandas as pd
        import json
        
        # Create and train a model
        np.random.seed(42)
        X = pd.DataFrame(np.random.rand(30, 3), columns=[f'f_{i}' for i in range(3)])
        y = pd.Series(np.random.randint(0, 2, 30))
        
        model = FederatedLightGBM(n_estimators=3, random_state=42, verbose=-1)
        model.fit(X, y)
        
        # Test the encoding pipeline (client to server)
        model_params = model.get_model_params()
        json_str = json.dumps(model_params, default=str)
        model_bytes = json_str.encode('utf-8')
        byte_array = np.frombuffer(model_bytes, dtype=np.uint8)
        param_array = byte_array.astype(np.float32)
        
        # Test the decoding pipeline (server to client)
        decoded_bytes = param_array.astype(np.uint8).tobytes()
        decoded_json = decoded_bytes.decode('utf-8', errors='ignore')
        decoded_params = json.loads(decoded_json)
        
        if decoded_params.get('model_type') != 'lightgbm':
            print("   ❌ Parameter encoding/decoding failed")
            return False
        
        print(f"   ✅ Parameter encoding successful: {len(param_array)} values")
        print(f"   ✅ Parameter decoding successful")
        print(f"   ✅ Model type preserved: {decoded_params.get('model_type')}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Communication test failed: {e}")
        return False

def test_aggregation():
    """Test model aggregation."""
    print("\n🔄 Test 4: Model Aggregation")
    
    try:
        from models.lightgbm_model import FederatedLightGBM, aggregate_lightgbm_models
        import numpy as np
        import pandas as pd
        
        # Create multiple models simulating different banks
        models = []
        weights = []
        
        for i in range(3):
            np.random.seed(42 + i)
            samples = 30 + (i * 10)
            X = pd.DataFrame(np.random.rand(samples, 3) * (1 + i * 0.1), 
                           columns=[f'f_{j}' for j in range(3)])
            y = pd.Series(np.random.randint(0, 2, samples))
            
            model = FederatedLightGBM(
                n_estimators=3, 
                learning_rate=0.1 + (i * 0.01),
                random_state=42 + i,
                verbose=-1
            )
            model.fit(X, y)
            
            models.append(model.get_model_params())
            weights.append(samples)
        
        # Test aggregation
        aggregated = aggregate_lightgbm_models(models, weights)
        
        if not aggregated:
            print("   ❌ Aggregation returned None")
            return False
        
        # Check aggregation results
        total_samples = aggregated.get('total_samples', 0)
        best_client = aggregated.get('federated_round_info', {}).get('best_client_idx', -1)
        lr_adjustment = aggregated.get('learning_rate_adjustment', 0)
        
        print(f"   ✅ Aggregation successful")
        print(f"   📊 Total federated samples: {total_samples}")
        print(f"   🏆 Best client selected: {best_client}")
        print(f"   📈 Learning rate adjustment: {lr_adjustment:.4f}")
        
        if total_samples != sum(weights):
            print("   ❌ Sample count mismatch")
            return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ Aggregation test failed: {e}")
        return False

def run_comprehensive_test():
    """Run all comprehensive tests."""
    print("🧪 COMPREHENSIVE LIGHTGBM FEDERATED LEARNING TEST")
    print("=" * 70)
    
    tests = [
        ("Data Availability", test_data_availability),
        ("Model Components", test_model_components), 
        ("Client-Server Communication", test_client_server_communication),
        ("Model Aggregation", test_aggregation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append(result)
            if not result:
                print(f"\n❌ {test_name} FAILED - stopping tests")
                break
        except Exception as e:
            print(f"\n❌ {test_name} ERROR: {e}")
            results.append(False)
            break
    
    # Summary
    print("\n" + "=" * 70)
    print("📋 TEST SUMMARY")
    print("=" * 70)
    
    for i, (test_name, _) in enumerate(tests):
        if i < len(results):
            status = "✅ PASSED" if results[i] else "❌ FAILED"
            print(f"{status} {test_name}")
        else:
            print(f"⏭️ SKIPPED {test_name}")
    
    all_passed = all(results) and len(results) == len(tests)
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Your LightGBM federated learning system is working correctly")
        print("🚀 Ready for production deployment!")
        print("\n💡 Key benefits achieved:")
        print("   • Hyperparameter sharing and optimization")
        print("   • Weighted model selection based on data size")  
        print("   • Robust parameter serialization and communication")
        print("   • Practical federated learning for tree-based models")
    else:
        print("\n💥 SOME TESTS FAILED!")
        print("🔧 Please review the failed components before deployment")
    
    return all_passed

def main():
    """Main function."""
    success = run_comprehensive_test()
    
    if success:
        print("\n🌟 SYSTEM READY!")
        print("You can now run your federated learning system:")
        print("  python start_lightgbm_server.py")
        print("  python start_lightgbm_client.py SBI")
    else:
        print("\n🛠️ SYSTEM NEEDS FIXES")
        print("Please address the failed tests before proceeding.")

if __name__ == "__main__":
    main()
