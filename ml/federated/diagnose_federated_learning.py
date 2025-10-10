#!/usr/bin/env python3
"""
LightGBM Federated Learning Diagnostic

Tests the federated learning aggregation process to identify issues.
"""

import sys
import os
import pickle
import json
import lightgbm as lgb
import numpy as np
import pandas as pd
from datetime import datetime

# Add paths
sys.path.append('.')
sys.path.append('./models')
sys.path.append('./config')

def load_and_inspect_model(model_path):
    """Load and inspect a LightGBM model file."""
    print(f"\n🔍 Inspecting Model: {model_path}")
    print("-" * 50)
    
    if not os.path.exists(model_path):
        print("❌ Model file does not exist")
        return None
        
    try:
        file_size = os.path.getsize(model_path)
        print(f"📁 File Size: {file_size:,} bytes ({file_size/1024:.2f} KB)")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            
        print(f"📊 Data Type: {type(model_data)}")
        
        if isinstance(model_data, dict):
            print("🔧 Dictionary Contents:")
            for key, value in model_data.items():
                if key == 'model_dump':
                    print(f"  {key}: {len(str(value))} characters")
                elif isinstance(value, (list, dict)):
                    print(f"  {key}: {type(value)} (length: {len(value)})")
                else:
                    print(f"  {key}: {value}")
                    
            # Check model dump quality
            model_dump = model_data.get('model_dump', '')
            if model_dump:
                print(f"\n🌳 Model Dump Analysis:")
                print(f"  Length: {len(model_dump)} characters")
                print(f"  Contains 'Tree': {'Tree' in model_dump}")
                print(f"  Contains 'split_feature': {'split_feature' in model_dump}")
                print(f"  First 200 chars: {model_dump[:200]}...")
                
                # Try to load as booster
                try:
                    booster = lgb.Booster(model_str=model_dump)
                    print(f"  ✅ Valid LightGBM model: {booster.num_trees()} trees")
                except Exception as e:
                    print(f"  ❌ Invalid LightGBM model: {e}")
            else:
                print("⚠️  No model_dump found")
                
        elif hasattr(model_data, '__class__'):
            print(f"📊 Model Class: {model_data.__class__}")
            if hasattr(model_data, 'booster_'):
                print(f"🌳 Has Booster: {model_data.booster_ is not None}")
                if model_data.booster_:
                    print(f"🌳 Number of Trees: {model_data.booster_.num_trees()}")
        
        return model_data
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def test_aggregation_function():
    """Test the LightGBM aggregation function directly."""
    print("\n🧪 Testing LightGBM Aggregation Function")
    print("=" * 60)
    
    try:
        from models.lightgbm_model import aggregate_lightgbm_models, FederatedLightGBM
        
        # Create mock model parameters (simulating multiple clients)
        mock_params_1 = {
            'model_dump': 'tree\n0 1 0.5 1 2 0 0.1\nsplit_feature=0\nthreshold=0.5',
            'params': {'learning_rate': 0.05, 'n_estimators': 50},
            'total_samples': 1000,
            'num_trees': 50,
            'model_type': 'lightgbm',
            'client_samples': 1000
        }
        
        mock_params_2 = {
            'model_dump': 'tree\n0 1 0.3 1 2 0 0.2\nsplit_feature=1\nthreshold=0.3',
            'params': {'learning_rate': 0.06, 'n_estimators': 60},
            'total_samples': 800,
            'num_trees': 60,
            'model_type': 'lightgbm',
            'client_samples': 800
        }
        
        model_params_list = [mock_params_1, mock_params_2]
        weights = [1000, 800]
        
        print("📥 Input Parameters:")
        print(f"  Client 1: {mock_params_1['total_samples']} samples, {mock_params_1['num_trees']} trees")
        print(f"  Client 2: {mock_params_2['total_samples']} samples, {mock_params_2['num_trees']} trees")
        
        # Test aggregation
        result = aggregate_lightgbm_models(model_params_list, weights)
        
        if result:
            print("✅ Aggregation Function Works!")
            print(f"📊 Result type: {type(result)}")
            print(f"🔧 Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
            
            if isinstance(result, dict):
                print(f"  Total samples: {result.get('total_samples', 'N/A')}")
                print(f"  Participating clients: {result.get('participating_clients', 'N/A')}")
                print(f"  Aggregation method: {result.get('aggregation_method', 'N/A')}")
                
                # Check if model_dump is preserved
                if 'model_dump' in result:
                    print(f"  Model dump length: {len(result['model_dump'])}")
                else:
                    print("  ⚠️  No model_dump in result")
        else:
            print("❌ Aggregation function returned None")
            
    except Exception as e:
        print(f"❌ Aggregation test failed: {e}")
        import traceback
        traceback.print_exc()

def check_serialization_process():
    """Check if the JSON serialization process is working correctly."""
    print("\n🔄 Testing Serialization Process")
    print("=" * 60)
    
    try:
        # Create a sample model state
        sample_state = {
            'model_dump': 'tree\nversion=v3\nnum_class=1\nnum_tree_per_iteration=1\nlabel_index=0\nmax_feature_idx=4\nobjective=binary sigmoid:1\nfeature_names=feature_0 feature_1 feature_2 feature_3 feature_4\nfeature_infos=none none none none none\ntree_sizes=100\n\nTree=0\nnum_leaves=31\nnum_cat=0\nsplit_feature=0 1 2 3 4\nsplit_gain=10.5 8.2 6.1 4.3 2.1\nthreshold=0.5 0.3 0.7 0.2 0.8',
            'params': {'learning_rate': 0.05, 'n_estimators': 200},
            'total_samples': 5000,
            'num_trees': 200,
            'model_type': 'lightgbm',
            'feature_names': ['amount', 'velocity_score', 'geo_anomaly_score', 'time_since_last', 'merchant_cat']
        }
        
        print("📦 Original Model State:")
        print(f"  Model dump length: {len(sample_state['model_dump'])}")
        print(f"  Total samples: {sample_state['total_samples']}")
        print(f"  Number of trees: {sample_state['num_trees']}")
        
        # Test JSON serialization (as done in federated server)
        json_str = json.dumps(sample_state, default=str)
        print(f"\n🔄 JSON Serialization:")
        print(f"  JSON length: {len(json_str)} characters")
        
        # Test encoding to bytes
        model_bytes = json_str.encode('utf-8')
        print(f"  Byte length: {len(model_bytes)} bytes")
        
        # Test conversion to float32 array (Flower format)
        byte_array = np.frombuffer(model_bytes, dtype=np.uint8)
        param_array = byte_array.astype(np.float32)
        print(f"  Float32 array length: {len(param_array)}")
        
        # Test reverse process (decoding)
        decoded_byte_array = param_array.astype(np.uint8)
        decoded_model_bytes = decoded_byte_array.tobytes()
        decoded_json_str = decoded_model_bytes.decode('utf-8', errors='ignore')
        
        print(f"\n🔄 Deserialization:")
        print(f"  Decoded JSON length: {len(decoded_json_str)} characters")
        
        if len(decoded_json_str) > 10:
            try:
                decoded_state = json.loads(decoded_json_str)
                print("  ✅ JSON deserialization successful")
                print(f"  Decoded model dump length: {len(decoded_state.get('model_dump', ''))}")
                print(f"  Decoded total samples: {decoded_state.get('total_samples', 'N/A')}")
            except json.JSONDecodeError as e:
                print(f"  ❌ JSON deserialization failed: {e}")
        else:
            print("  ❌ Decoded JSON too short")
            
    except Exception as e:
        print(f"❌ Serialization test failed: {e}")
        import traceback
        traceback.print_exc()

def check_model_saving():
    """Test the model saving process."""
    print("\n💾 Testing Model Saving Process")
    print("=" * 60)
    
    try:
        # Create sample aggregated parameters
        aggregated_params = {
            'model_dump': 'tree\nversion=v3\nnum_class=1\nnum_tree_per_iteration=1\nlabel_index=0\nmax_feature_idx=10\nobjective=binary sigmoid:1\nfeature_names=amount velocity_score geo_anomaly_score\nfeature_infos=none none none\ntree_sizes=200 180 160 140 120\n\nTree=0\nnum_leaves=31\nnum_cat=0\nsplit_feature=0 1 2\nsplit_gain=15.2 12.1 9.8\nthreshold=1000.0 5.5 2.3\ninternal_value=0.1 0.2 0.3\ninternal_weight=100 80 60\ninternal_count=500 300 200\nleaf_value=0.5 -0.3 0.8 -0.1\nleaf_weight=50 40 30 20\nleaf_count=50 40 30 20',
            'params': {'learning_rate': 0.05, 'n_estimators': 200, 'max_depth': 10},
            'total_samples': 10000,
            'num_trees': 200,
            'model_type': 'lightgbm',
            'participating_clients': 3,
            'round': 5
        }
        
        print("📊 Sample Aggregated Parameters:")
        print(f"  Model dump length: {len(aggregated_params['model_dump'])}")
        print(f"  Total samples: {aggregated_params['total_samples']}")
        print(f"  Trees: {aggregated_params['num_trees']}")
        
        # Test saving process (as done in server)
        test_save_path = "./test_saved_model.pkl"
        
        model_data = {
            'model_dump': aggregated_params.get('model_dump', ''),
            'round': aggregated_params.get('round', 1),
            'total_samples': aggregated_params.get('total_samples', 0),
            'last_updated': aggregated_params.get('round', 1),
            'version_info': {
                'created_at': datetime.now().isoformat(),
                'federated_round': aggregated_params.get('round', 1),
                'model_type': 'lightgbm_federated'
            }
        }
        
        with open(test_save_path, 'wb') as f:
            pickle.dump(model_data, f)
            
        # Check saved file
        saved_size = os.path.getsize(test_save_path)
        print(f"\n💾 Saved Model:")
        print(f"  File size: {saved_size:,} bytes ({saved_size/1024:.2f} KB)")
        
        # Load it back
        with open(test_save_path, 'rb') as f:
            loaded_data = pickle.load(f)
            
        print(f"  Loaded model dump length: {len(loaded_data.get('model_dump', ''))}")
        print(f"  Loaded total samples: {loaded_data.get('total_samples', 'N/A')}")
        
        # Clean up
        os.unlink(test_save_path)
        
        if saved_size > 50000:  # > 50KB
            print("  ✅ Saved model size looks good")
        else:
            print("  ⚠️  Saved model size seems too small")
            
    except Exception as e:
        print(f"❌ Model saving test failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main diagnostic function."""
    print("🚀 LightGBM Federated Learning Diagnostic")
    print("=" * 80)
    
    # Check current models
    load_and_inspect_model("./trained_models/lightgbm_model.pkl")
    load_and_inspect_model("./trained_models/latest_lightgbm_federated.pkl")
    
    # Test aggregation
    test_aggregation_function()
    
    # Test serialization
    check_serialization_process()
    
    # Test model saving
    check_model_saving()
    
    print("\n" + "=" * 80)
    print("🏁 Diagnostic Complete!")
    print("\n💡 If your federated model becomes small (4KB), the issue is likely:")
    print("   1. Empty or corrupted model_dump during aggregation")
    print("   2. JSON serialization/deserialization corruption") 
    print("   3. Model saving process losing the actual tree data")

if __name__ == "__main__":
    main()
