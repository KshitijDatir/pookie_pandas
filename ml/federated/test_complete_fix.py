#!/usr/bin/env python3
"""
Final test to confirm the complete federated flow produces large models, not 4KB.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pickle
import json
import logging
from lightgbm_federated_server import LightGBMFederatedStrategy
from models.lightgbm_model import FederatedLightGBM, aggregate_lightgbm_models
import flwr as fl
import numpy as np

def create_mock_client_fit_results():
    """Create mock client FitRes objects like in real federated learning."""
    
    # Mock FitRes class (simplified)
    class MockFitRes:
        def __init__(self, parameters, num_examples, metrics=None):
            self.parameters = parameters
            self.num_examples = num_examples
            self.metrics = metrics or {}
    
    # Create small client models
    client_results = []
    for i in range(2):  # 2 clients
        
        # Create small client model  
        from sklearn.datasets import make_classification
        import pandas as pd
        
        X, y = make_classification(n_samples=30, n_features=10, random_state=42+i)
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        
        model = FederatedLightGBM()
        model.model_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'num_boost_round': 30,  # Small model
            'learning_rate': 0.1,
            'verbose': -1
        }
        model.fit(df, y)
        
        # Get model parameters
        model_params = model.get_model_params()
        model_params['client_id'] = f'client_{i}'
        
        # Convert to Flower parameters format
        json_str = json.dumps(model_params, default=str)
        model_bytes = json_str.encode('utf-8')
        byte_array = np.frombuffer(model_bytes, dtype=np.uint8)
        param_array = byte_array.astype(np.float32)
        parameters = fl.common.ndarrays_to_parameters([param_array])
        
        # Create mock FitRes
        fit_res = MockFitRes(
            parameters=parameters,
            num_examples=30,
            metrics={"accuracy": 0.85 + (i * 0.05)}
        )
        
        client_results.append(fit_res)
        print(f"✅ Mock client {i+1}: 30 samples, {model_params.get('num_trees', 0)} trees")
    
    return client_results

def test_complete_federated_round():
    """Test complete federated round with server strategy."""
    
    logging.basicConfig(level=logging.INFO)
    print("🚀 Testing Complete Federated Round (4KB Fix)")
    
    # Step 1: Initialize server with large base model
    base_model_path = "trained_models/lightgbm_model.pkl"
    strategy = LightGBMFederatedStrategy(base_model_path=base_model_path)
    
    print(f"\n📊 Server initialized with base model:")
    if strategy.base_model:
        params = strategy.base_model.get_model_params()
        print(f"   - Trees: {params.get('num_trees', 0)}")
        print(f"   - Samples: {params.get('total_samples', 0):,}")
        print(f"   - Model size: {len(params.get('model_dump', ''))} chars")
    
    # Step 2: Create mock client results
    print(f"\n👥 Creating mock client training results...")
    client_results = create_mock_client_fit_results()
    
    # Step 3: Test server aggregation (this is where the 4KB bug was)
    print(f"\n🔄 Testing server aggregation...")
    
    try:
        # This calls the fixed aggregate_fit method
        aggregated_params, metrics = strategy.aggregate_fit(
            server_round=1,
            results=client_results,
            failures=[]
        )
        
        print(f"✅ Aggregation completed")
        print(f"📊 Metrics: {metrics}")
        
        # Step 4: Test what gets saved (decode the aggregated parameters)
        print(f"\n💾 Testing saved model size...")
        
        if aggregated_params:
            # Decode the aggregated parameters like the server would
            param_arrays = fl.common.parameters_to_ndarrays(aggregated_params)
            if len(param_arrays) > 0:
                byte_array = param_arrays[0].astype(np.uint8)
                model_bytes = byte_array.tobytes()
                json_str = model_bytes.decode('utf-8', errors='ignore')
                
                if len(json_str) > 100:
                    model_data = json.loads(json_str)
                    
                    # Test model data
                    total_samples = model_data.get('total_samples', 0)
                    num_trees = model_data.get('num_trees', 0)
                    model_dump_size = len(model_data.get('model_dump', ''))
                    
                    print(f"📊 Aggregated model:")
                    print(f"   - Trees: {num_trees}")
                    print(f"   - Total samples: {total_samples:,}")
                    print(f"   - Model dump size: {model_dump_size} chars")
                    
                    # Create a test save to check file size
                    test_save_data = {
                        'model_dump': model_data.get('model_dump', ''),
                        'total_samples': total_samples,
                        'round': 1
                    }
                    
                    test_path = "test_aggregated_model.pkl"
                    with open(test_path, 'wb') as f:
                        pickle.dump(test_save_data, f)
                    
                    file_size = os.path.getsize(test_path)
                    file_size_kb = file_size / 1024
                    
                    print(f"💾 Saved model file size: {file_size_kb:.1f} KB")
                    
                    # The key test: Is it still 4KB or is it large?
                    if file_size_kb < 10:
                        print(f"❌ STILL BROKEN: Model is tiny ({file_size_kb:.1f} KB)")
                        print(f"   - This indicates the global model is not being used correctly")
                    elif file_size_kb > 100:
                        print(f"🎉 SUCCESS: Large model preserved ({file_size_kb:.1f} KB)")
                        print(f"   - Global model (4M samples) is being used correctly!")
                        print(f"   - The 4KB bug is FIXED!")
                    else:
                        print(f"⚠️ MEDIUM: Model size is {file_size_kb:.1f} KB (check if this is expected)")
                    
                    # Cleanup
                    if os.path.exists(test_path):
                        os.remove(test_path)
                    
                else:
                    print(f"❌ Failed to decode aggregated parameters (too short)")
            else:
                print(f"❌ No parameter arrays in aggregated parameters")
        else:
            print(f"❌ Aggregation returned no parameters")
            
    except Exception as e:
        print(f"❌ Aggregation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_complete_federated_round()
