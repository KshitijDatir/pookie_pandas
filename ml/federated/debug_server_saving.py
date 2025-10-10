#!/usr/bin/env python3
"""
Debug script to test what gets saved by the server save_global_model method.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pickle
import json
import logging
from datetime import datetime
from models.lightgbm_model import FederatedLightGBM, aggregate_lightgbm_models

def create_mock_large_model():
    """Create a mock large model like the 4M sample global model."""
    # Create and train a model to get a realistic model_dump
    from sklearn.datasets import make_classification
    import pandas as pd
    
    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    
    model = FederatedLightGBM()
    model.model_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'num_boost_round': 100,
        'learning_rate': 0.1,
        'num_leaves': 31,
        'verbose': -1
    }
    model.fit(df, y)
    
    params = model.get_model_params()
    
    # Simulate it being a 4M sample model
    params['total_samples'] = 4000000
    params['global_model'] = True
    
    return params

def create_mock_client_models():
    """Create mock small client models."""
    client_models = []
    
    for i in range(3):
        from sklearn.datasets import make_classification
        import pandas as pd
        
        X, y = make_classification(n_samples=50, n_features=10, random_state=42+i)
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        
        model = FederatedLightGBM()
        model.model_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'num_boost_round': 50,
            'learning_rate': 0.1 + (i * 0.01),
            'num_leaves': 25,
            'verbose': -1
        }
        model.fit(df, y)
        
        params = model.get_model_params()
        params['client_id'] = f'client_{i}'
        
        client_models.append(params)
    
    return client_models

def mock_save_global_model(model_params: dict, save_path: str, round_num: int):
    """Mock the server's save_global_model method to debug what gets saved."""
    
    print(f"\n🔍 DEBUGGING save_global_model:")
    print(f"📊 Input model_params keys: {list(model_params.keys())}")
    print(f"📊 Total samples in params: {model_params.get('total_samples', 'NOT_FOUND')}")
    print(f"📊 Model dump length: {len(model_params.get('model_dump', ''))}")
    print(f"📊 Num trees: {model_params.get('num_trees', 'NOT_FOUND')}")
    
    # Replicate the exact server logic
    model_data = {
        'model_dump': model_params.get('model_dump', ''),
        'round': round_num,
        'total_samples': model_params.get('total_samples', 0),  # This is the suspected bug
        'last_updated': round_num,
        'version_info': {
            'created_at': datetime.now().isoformat(),
            'federated_round': round_num,
            'model_type': 'lightgbm_federated'
        }
    }
    
    print(f"🔍 What will be saved:")
    print(f"📊 model_data['total_samples']: {model_data['total_samples']}")
    print(f"📊 model_data['model_dump'] length: {len(model_data['model_dump'])}")
    
    # Save to debug file
    with open(save_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    file_size = os.path.getsize(save_path)
    print(f"✅ Saved debug model: {file_size/1024:.2f} KB")
    
    return model_data

def main():
    """Test the complete flow from aggregation to saving."""
    
    logging.basicConfig(level=logging.INFO)
    
    print("🚀 Debugging Server Model Saving")
    
    # Step 1: Create mock models
    print("\n📊 Creating mock models...")
    global_model = create_mock_large_model()
    client_models = create_mock_client_models()
    weights = [10, 50, 100]  # Small client weights
    
    print(f"✅ Global model: {global_model['total_samples']:,} samples, {len(global_model['model_dump'])} chars")
    print(f"✅ Client models: {len(client_models)} models")
    
    # Step 2: Test aggregation
    print("\n🔄 Testing aggregation...")
    aggregated_params = aggregate_lightgbm_models(client_models, weights, global_model)
    
    print(f"📊 Aggregated total_samples: {aggregated_params.get('total_samples', 'NOT_FOUND')}")
    print(f"📊 Aggregated model_dump length: {len(aggregated_params.get('model_dump', ''))}")
    print(f"📊 Aggregated num_trees: {aggregated_params.get('num_trees', 'NOT_FOUND')}")
    
    # Step 3: Test saving
    print("\n💾 Testing server save...")
    debug_path = "debug_saved_model.pkl"
    saved_data = mock_save_global_model(aggregated_params, debug_path, 1)
    
    # Step 4: Verify what was actually saved
    print("\n🔍 Verifying saved model...")
    with open(debug_path, 'rb') as f:
        loaded_data = pickle.load(f)
    
    print(f"📊 Loaded total_samples: {loaded_data.get('total_samples', 'NOT_FOUND')}")
    print(f"📊 Loaded model_dump length: {len(loaded_data.get('model_dump', ''))}")
    
    # Cleanup
    if os.path.exists(debug_path):
        os.remove(debug_path)

if __name__ == "__main__":
    main()
