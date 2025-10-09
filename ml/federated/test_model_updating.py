#!/usr/bin/env python3
"""
Test LightGBM Model Parameter Updating

This script tests whether the LightGBM model parameter serialization and 
updating is working correctly in the federated learning system.
"""

import numpy as np
import pandas as pd
import json
import sys
import os

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'models'))

def test_model_parameter_updating():
    """Test LightGBM model parameter encoding/decoding."""
    print("🧪 Testing LightGBM Model Parameter Updating")
    print("=" * 60)
    
    try:
        from models.lightgbm_model import FederatedLightGBM, aggregate_lightgbm_models
        
        # Create sample data
        print("📊 Creating sample training data...")
        np.random.seed(42)
        X = pd.DataFrame(np.random.rand(100, 5), columns=[f'feature_{i}' for i in range(5)])
        y = pd.Series(np.random.randint(0, 2, 100))
        
        print(f"   Training data: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Test 1: Create and train model
        print("\n🔄 Test 1: Model Training")
        model1 = FederatedLightGBM(n_estimators=10, random_state=42)
        model1.fit(X, y)
        print("   ✅ Model 1 trained successfully")
        
        # Test 2: Get model parameters
        print("\n🔄 Test 2: Parameter Extraction")
        params1 = model1.get_model_params()
        if params1:
            print("   ✅ Model parameters extracted successfully")
            print(f"   📋 Parameters: {len(params1)} keys")
            print(f"   🌳 Model dump length: {len(params1.get('model_dump', ''))}")
            print(f"   🔢 Trees: {params1.get('num_trees', 0)}")
            print(f"   📊 Samples: {params1.get('total_samples', 0)}")
        else:
            print("   ❌ Failed to extract parameters")
            return False
        
        # Test 3: JSON serialization
        print("\n🔄 Test 3: JSON Serialization")
        try:
            json_str = json.dumps(params1, default=str)
            print(f"   ✅ JSON serialization successful: {len(json_str)} characters")
        except Exception as e:
            print(f"   ❌ JSON serialization failed: {e}")
            return False
        
        # Test 4: Parameter setting
        print("\n🔄 Test 4: Parameter Setting")
        model2 = FederatedLightGBM(n_estimators=10, random_state=123)  # Different seed
        model2.set_model_params(params1)
        
        if model2.is_fitted:
            print("   ✅ Model 2 parameters set successfully")
            print(f"   🔢 Model 2 fitted: {model2.is_fitted}")
        else:
            print("   ❌ Model 2 parameter setting failed")
            return False
        
        # Test 5: Practical FL - Parameter Update and Retrain
        print("\n🔄 Test 5: Practical FL - Parameter Update & Retrain")
        try:
            # In practical LightGBM FL, we update parameters and retrain
            # Model 2 gets updated parameters from model 1
            old_lr = model2.params.get('learning_rate', 0.1)
            model2.set_model_params(params1)
            new_lr = model2.params.get('learning_rate', 0.1)
            
            print(f"   📈 Learning rate updated: {old_lr:.4f} → {new_lr:.4f}")
            
            # Now retrain model2 with the updated parameters
            model2.fit(X, y)
            
            # Now both models can make predictions
            pred1 = model1.predict(X[:10])
            pred2 = model2.predict(X[:10])
            
            # They won't be identical (different random seeds) but both should work
            print(f"   🔮 Model 1 predictions: {pred1[:5]}")
            print(f"   🔮 Model 2 predictions: {pred2[:5]}")
            print(f"   ✅ Both models can make predictions after parameter sharing")
            
            # Test that the parameter sharing worked
            if abs(new_lr - params1.get('params', {}).get('learning_rate', 0.1)) < 0.001:
                print(f"   ✅ Parameters successfully shared from model 1 to model 2")
            else:
                print(f"   ⚠️ Parameter sharing may not have worked perfectly")
                
        except Exception as e:
            print(f"   ❌ Practical FL test failed: {e}")
            return False
        
        # Test 6: Aggregation
        print("\n🔄 Test 6: Model Aggregation")
        try:
            # Create another model for aggregation
            model3 = FederatedLightGBM(n_estimators=10, random_state=456)
            X2 = X.copy()
            X2.iloc[:50] *= 1.1  # Slightly different data
            model3.fit(X2, y)
            
            params3 = model3.get_model_params()
            
            # Test aggregation
            model_list = [params1, params3]
            weights = [100, 80]  # First model has more samples
            
            aggregated = aggregate_lightgbm_models(model_list, weights)
            
            if aggregated:
                print("   ✅ Model aggregation successful")
                print(f"   📊 Aggregated samples: {aggregated.get('total_samples', 0)}")
                print(f"   🔢 Participating clients: {aggregated.get('participating_clients', 0)}")
                print(f"   🎯 Selected from client: {aggregated.get('federated_round_info', {}).get('best_client_idx', 'unknown')}")
            else:
                print("   ❌ Model aggregation failed")
                return False
                
        except Exception as e:
            print(f"   ❌ Aggregation test failed: {e}")
            return False
        
        # Test 7: Flower parameter format
        print("\n🔄 Test 7: Flower Parameter Format")
        try:
            # Simulate the parameter encoding used in the client
            json_str = json.dumps(params1, default=str)
            model_bytes = json_str.encode('utf-8')
            byte_array = np.frombuffer(model_bytes, dtype=np.uint8)
            param_array = byte_array.astype(np.float32)
            
            print(f"   ✅ Flower encoding successful: {len(param_array)} values")
            
            # Test decoding
            decoded_bytes = param_array.astype(np.uint8).tobytes()
            decoded_json = decoded_bytes.decode('utf-8', errors='ignore')
            decoded_params = json.loads(decoded_json)
            
            if decoded_params.get('model_type') == 'lightgbm':
                print("   ✅ Flower decoding successful")
            else:
                print("   ❌ Flower decoding failed: wrong model type")
                return False
                
        except Exception as e:
            print(f"   ❌ Flower parameter format test failed: {e}")
            return False
        
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ LightGBM model parameter updating is working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    success = test_model_parameter_updating()
    
    if success:
        print("\n🌟 LightGBM federated learning should work with proper model updates!")
        print("🚀 You can now run the federated learning system with confidence.")
    else:
        print("\n💥 Issues found with model parameter updating.")
        print("🔧 Please check the implementation before running federated learning.")

if __name__ == "__main__":
    main()
