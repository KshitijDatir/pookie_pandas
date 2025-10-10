#!/usr/bin/env python3
"""
Live Federated Learning Diagnostic

This will trace exactly what happens during model parameter exchange
to find where the 4KB models are coming from.
"""

import sys
import os
import pickle
import json
import numpy as np
import lightgbm as lgb
from datetime import datetime

# Add paths
sys.path.append('./models')
sys.path.append('./config')

def inspect_current_broken_model():
    """Inspect the current broken 3.4KB model to understand what went wrong."""
    print("🔍 Inspecting Current Broken Model")
    print("=" * 60)
    
    current_path = "./trained_models/latest_lightgbm_federated.pkl"
    
    if not os.path.exists(current_path):
        print("❌ Current model file not found")
        return None
    
    try:
        with open(current_path, 'rb') as f:
            broken_model = pickle.load(f)
        
        file_size = os.path.getsize(current_path)
        print(f"📁 File size: {file_size:,} bytes ({file_size/1024:.2f} KB)")
        print(f"📊 Data type: {type(broken_model)}")
        
        if isinstance(broken_model, dict):
            print("🔧 Dictionary contents:")
            for key, value in broken_model.items():
                if key == 'model_dump':
                    model_dump = str(value)
                    print(f"  {key}: {len(model_dump)} characters")
                    if len(model_dump) > 0:
                        print(f"    First 200 chars: {model_dump[:200]}")
                        
                        # Try to validate the model dump
                        if len(model_dump) < 1000:
                            print(f"    🚨 Model dump is suspiciously short! ({len(model_dump)} chars)")
                        else:
                            try:
                                booster = lgb.Booster(model_str=model_dump)
                                print(f"    ✅ Valid LightGBM: {booster.num_trees()} trees")
                            except Exception as e:
                                print(f"    ❌ Invalid model dump: {e}")
                    else:
                        print(f"    🚨 EMPTY model dump!")
                        
                elif isinstance(value, (list, dict)):
                    print(f"  {key}: {type(value)} (length: {len(value)})")
                else:
                    print(f"  {key}: {value}")
        
        elif hasattr(broken_model, '__class__'):
            print(f"📊 Object type: {broken_model.__class__}")
            
            # Check if it's a direct LightGBM model
            if hasattr(broken_model, 'booster_') and broken_model.booster_:
                print(f"🌳 Has booster: {broken_model.booster_.num_trees()} trees")
                
        return broken_model
        
    except Exception as e:
        print(f"❌ Error inspecting broken model: {e}")
        return None

def trace_parameter_serialization():
    """Test the parameter serialization process that might be causing issues."""
    print("\n🔄 Tracing Parameter Serialization Process")
    print("=" * 60)
    
    try:
        # Load a good model
        original_path = "./trained_models/lightgbm_model.pkl"
        with open(original_path, 'rb') as f:
            good_model = pickle.load(f)
            
        print(f"✅ Loaded original model: {good_model.booster_.num_trees()} trees")
        
        # Create FederatedLightGBM and extract parameters (simulating client)
        from models.lightgbm_model import FederatedLightGBM
        
        fed_model = FederatedLightGBM(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=10,
            num_leaves=31,
            random_state=42
        )
        
        # Manually load the model (simulating aggregation from server)
        model_dump = good_model.booster_.model_to_string()
        print(f"📤 Original model dump: {len(model_dump)} characters")
        
        # Simulate set_model_params
        test_params = {
            'model_dump': model_dump,
            'params': good_model.get_params(),
            'total_samples': 10000,
            'model_type': 'lightgbm'
        }
        
        fed_model.set_model_params(test_params)
        print(f"🔄 Model loaded: {fed_model.is_fitted}, Trees: {fed_model.model._Booster.num_trees() if fed_model.is_fitted else 'N/A'}")
        
        # Now extract parameters (simulating get_parameters for server)
        extracted_params = fed_model.get_model_params()
        
        if extracted_params:
            extracted_dump = extracted_params.get('model_dump', '')
            print(f"📥 Extracted model dump: {len(extracted_dump)} characters")
            
            if len(extracted_dump) != len(model_dump):
                print(f"🚨 SIZE MISMATCH! Original: {len(model_dump)}, Extracted: {len(extracted_dump)}")
                print("🚨 This could be the source of the problem!")
                
                # Compare first parts
                print(f"Original start: {model_dump[:200]}")
                print(f"Extracted start: {extracted_dump[:200]}")
                
            else:
                print("✅ Model dump sizes match")
                
            # Test the Flower serialization process
            print("\n🔄 Testing Flower Serialization...")
            
            # Simulate what happens in get_parameters()
            model_state = {
                'model_dump': extracted_params.get('model_dump', ''),
                'params': extracted_params.get('params', {}),
                'feature_names': extracted_params.get('feature_names'),
                'n_features': extracted_params.get('n_features'),
                'total_samples': extracted_params.get('total_samples', 0),
                'num_trees': extracted_params.get('num_trees', 0),
                'model_type': 'lightgbm'
            }
            
            # Convert to JSON
            json_str = json.dumps(model_state, default=str)
            print(f"📦 JSON serialization: {len(json_str)} characters")
            
            # Convert to bytes
            model_bytes = json_str.encode('utf-8')
            print(f"📦 Bytes: {len(model_bytes)} bytes")
            
            # Convert to float32 array (Flower format)
            byte_array = np.frombuffer(model_bytes, dtype=np.uint8)
            param_array = byte_array.astype(np.float32)
            print(f"📦 Float32 array: {len(param_array)} elements")
            
            # Simulate reverse process (what set_parameters does)
            print("\n🔄 Testing Deserialization...")
            
            # Convert back
            decoded_byte_array = param_array.astype(np.uint8)
            decoded_model_bytes = decoded_byte_array.tobytes()
            decoded_json_str = decoded_model_bytes.decode('utf-8', errors='ignore')
            print(f"📥 Decoded JSON: {len(decoded_json_str)} characters")
            
            if len(decoded_json_str) != len(json_str):
                print(f"🚨 JSON CORRUPTION! Original: {len(json_str)}, Decoded: {len(decoded_json_str)}")
                print("🚨 This is likely causing the 4KB models!")
                
                # Find where corruption happens
                for i in range(min(len(json_str), len(decoded_json_str))):
                    if json_str[i] != decoded_json_str[i]:
                        print(f"🚨 First difference at position {i}:")
                        print(f"   Original: '{json_str[i-10:i+10]}'")
                        print(f"   Decoded:  '{decoded_json_str[i-10:i+10]}'")
                        break
            else:
                print("✅ JSON serialization/deserialization successful")
                
                # Try to parse the decoded JSON
                try:
                    decoded_state = json.loads(decoded_json_str)
                    decoded_dump = decoded_state.get('model_dump', '')
                    print(f"✅ Decoded model dump: {len(decoded_dump)} characters")
                    
                    if len(decoded_dump) == len(model_dump):
                        print("✅ Complete model dump preservation confirmed")
                    else:
                        print(f"🚨 Model dump size changed: {len(model_dump)} → {len(decoded_dump)}")
                        
                except json.JSONDecodeError as e:
                    print(f"❌ JSON decode error: {e}")
        
        else:
            print("❌ Failed to extract parameters from federated model")
            
    except Exception as e:
        print(f"❌ Error in parameter serialization test: {e}")
        import traceback
        traceback.print_exc()

def check_server_model_saving():
    """Check if the server model saving process is corrupting models."""
    print("\n💾 Testing Server Model Saving Process")
    print("=" * 60)
    
    try:
        # Load original model
        original_path = "./trained_models/lightgbm_model.pkl"
        with open(original_path, 'rb') as f:
            original_model = pickle.load(f)
            
        model_dump = original_model.booster_.model_to_string()
        
        # Simulate the server saving process
        aggregated_params = {
            'model_dump': model_dump,
            'params': original_model.get_params(),
            'total_samples': 10000,
            'num_trees': 200,
            'model_type': 'lightgbm'
        }
        
        # Simulate what server does in save_global_model
        model_data = {
            'model_dump': aggregated_params.get('model_dump', ''),
            'round': 1,
            'total_samples': aggregated_params.get('total_samples', 0),
            'last_updated': 1,
            'version_info': {
                'created_at': datetime.now().isoformat(),
                'federated_round': 1,
                'model_type': 'lightgbm_federated'
            }
        }
        
        print(f"📦 Server model data:")
        print(f"  Model dump: {len(model_data['model_dump'])} characters")
        print(f"  Total samples: {model_data['total_samples']}")
        
        # Test saving
        test_path = "./test_server_save.pkl"
        with open(test_path, 'wb') as f:
            pickle.dump(model_data, f)
            
        saved_size = os.path.getsize(test_path)
        print(f"💾 Saved file size: {saved_size:,} bytes ({saved_size/1024:.2f} KB)")
        
        # Test loading
        with open(test_path, 'rb') as f:
            loaded_data = pickle.load(f)
            
        loaded_dump = loaded_data.get('model_dump', '')
        print(f"📥 Loaded model dump: {len(loaded_dump)} characters")
        
        if len(loaded_dump) == len(model_dump):
            print("✅ Server saving/loading preserves model dump")
        else:
            print(f"🚨 Server saving corrupted model dump: {len(model_dump)} → {len(loaded_dump)}")
            
        # Clean up
        os.unlink(test_path)
        
        if saved_size < 50000:  # Less than 50KB
            print("🚨 Saved model is too small - this suggests server saving issue!")
            return False
        else:
            print("✅ Server model saving appears to work correctly")
            return True
            
    except Exception as e:
        print(f"❌ Error in server saving test: {e}")
        return False

def main():
    """Main diagnostic function."""
    print("🚀 Live Federated Learning Diagnostic")
    print("🎯 Finding the REAL cause of 4KB models")
    print("=" * 80)
    
    # Step 1: Inspect the current broken model
    broken_model = inspect_current_broken_model()
    
    # Step 2: Test parameter serialization
    trace_parameter_serialization()
    
    # Step 3: Test server model saving
    server_ok = check_server_model_saving()
    
    print("\n" + "=" * 80)
    print("🎯 DIAGNOSTIC SUMMARY")
    
    if broken_model is None:
        print("❌ Could not inspect broken model")
    elif isinstance(broken_model, dict):
        model_dump = broken_model.get('model_dump', '')
        if len(model_dump) < 1000:
            print(f"🚨 PROBLEM FOUND: Model dump is only {len(model_dump)} characters")
            print("🚨 This is why the model file is only 4KB!")
            print("💡 The issue is in model serialization/deserialization")
        else:
            print("✅ Model dump in file looks okay")
    
    print("\n💡 NEXT STEPS:")
    print("1. Check the federated learning logs during actual runs")
    print("2. Add debug prints to the model serialization process") 
    print("3. Monitor what happens to model_dump during aggregation")

if __name__ == "__main__":
    main()
