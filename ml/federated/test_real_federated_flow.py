#!/usr/bin/env python3
"""
Test the real federated flow to see what's actually happening in production.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pickle
import pandas as pd
import numpy as np
from models.lightgbm_model import FederatedLightGBM, aggregate_lightgbm_models

def create_realistic_base_model():
    """Create a realistic base model to simulate the 4M sample scenario."""
    print("🏭 Creating realistic base model (simulating 4M samples)...")
    
    # Create larger training dataset
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=3000,  # Larger for testing
        n_features=15, 
        n_informative=12,
        n_redundant=2,
        n_clusters_per_class=1,
        random_state=42
    )
    
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    
    # Create and train model
    model = FederatedLightGBM()
    model.model_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'num_boost_round': 150,  # More trees like production
        'learning_rate': 0.1,
        'max_depth': 8,
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'verbose': -1
    }
    
    model.fit(df, y)
    
    # Get model parameters and simulate 4M samples
    params = model.get_model_params()
    params['total_samples'] = 4000000  # Simulate real production scenario
    
    print(f"✅ Base model created: {params.get('num_trees', 0)} trees, {len(params.get('model_dump', ''))} chars")
    print(f"✅ Simulated total samples: {params['total_samples']:,}")
    
    return model, params

def save_base_model(model, params, filepath):
    """Save the base model like in production."""
    print(f"\n💾 Saving base model to {filepath}...")
    
    # Save as LGBMClassifier (like production)
    with open(filepath, 'wb') as f:
        pickle.dump(model.model, f)
    
    file_size = os.path.getsize(filepath)
    print(f"✅ Base model saved: {file_size/1024:.1f} KB")
    
    # Also save the parameters separately for debugging
    params_path = filepath.replace('.pkl', '_params.pkl')
    with open(params_path, 'wb') as f:
        pickle.dump(params, f)
    
    print(f"✅ Parameters saved: {params_path}")

def create_small_client_model():
    """Create a small client model like in production (10-100 samples)."""
    from sklearn.datasets import make_classification
    
    # Very small dataset like real clients
    X, y = make_classification(
        n_samples=25,  # Very small like production
        n_features=15,
        n_informative=10,
        n_redundant=2,
        random_state=np.random.randint(1000)
    )
    
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    
    model = FederatedLightGBM()
    model.model_params = {
        'objective': 'binary',
        'metric': 'binary_logloss', 
        'num_boost_round': 50,  # Fewer trees for small dataset
        'learning_rate': 0.1,
        'max_depth': 6,
        'num_leaves': 15,
        'verbose': -1
    }
    
    model.fit(df, y)
    return model.get_model_params()

def simulate_real_federated_round():
    """Simulate what actually happens in a federated round."""
    
    print("🚀 Simulating Real Federated Round")
    
    # Step 1: Create and save base model (like production setup)
    base_model_obj, base_model_params = create_realistic_base_model()
    base_model_path = "trained_models/lightgbm_model.pkl"
    
    # Ensure directory exists
    os.makedirs("trained_models", exist_ok=True)
    save_base_model(base_model_obj, base_model_params, base_model_path)
    
    # Step 2: Create small client models (like real clients)
    print(f"\n👥 Creating small client models...")
    client_params_list = []
    weights = []
    
    for i in range(3):
        client_params = create_small_client_model()
        client_samples = client_params.get('total_samples', 25)
        
        client_params_list.append(client_params)
        weights.append(client_samples)
        
        print(f"✅ Client {i+1}: {client_params.get('num_trees', 0)} trees, "
              f"{client_samples} samples, {len(client_params.get('model_dump', ''))} chars")
    
    total_client_samples = sum(weights)
    print(f"📊 Total client samples: {total_client_samples}")
    
    # Step 3: Test aggregation with and without global model
    print(f"\n🔄 Testing aggregation scenarios...")
    
    # Scenario 1: Without global model (current production issue)
    print(f"\n--- Scenario 1: Without Global Model ---")
    agg_without_global = aggregate_lightgbm_models(client_params_list, weights, None)
    print(f"📊 Result: {agg_without_global.get('num_trees', 0)} trees, "
          f"{agg_without_global.get('total_samples', 0)} samples, "
          f"{len(agg_without_global.get('model_dump', ''))} chars")
    
    # Scenario 2: With global model (fixed version)
    print(f"\n--- Scenario 2: With Global Model ---")
    agg_with_global = aggregate_lightgbm_models(client_params_list, weights, base_model_params)
    print(f"📊 Result: {agg_with_global.get('num_trees', 0)} trees, "
          f"{agg_with_global.get('total_samples', 0)} samples, "
          f"{len(agg_with_global.get('model_dump', ''))} chars")
    
    # Step 4: Test what gets saved
    print(f"\n💾 Testing save operations...")
    
    # Save without global (what's happening in production)
    save_without_path = "debug_without_global.pkl"
    save_data_without = {
        'model_dump': agg_without_global.get('model_dump', ''),
        'total_samples': agg_without_global.get('total_samples', 0),
        'round': 1
    }
    with open(save_without_path, 'wb') as f:
        pickle.dump(save_data_without, f)
    
    size_without = os.path.getsize(save_without_path)
    print(f"❌ Without global: {size_without/1024:.1f} KB")
    
    # Save with global (fixed version)
    save_with_path = "debug_with_global.pkl"
    save_data_with = {
        'model_dump': agg_with_global.get('model_dump', ''),
        'total_samples': agg_with_global.get('total_samples', 0),
        'round': 1
    }
    with open(save_with_path, 'wb') as f:
        pickle.dump(save_data_with, f)
    
    size_with = os.path.getsize(save_with_path)
    print(f"✅ With global: {size_with/1024:.1f} KB")
    
    # Cleanup
    for path in [save_without_path, save_with_path]:
        if os.path.exists(path):
            os.remove(path)
    
    print(f"\n🎯 DIAGNOSIS:")
    print(f"The issue is that federated server is not passing the global model to aggregation!")
    print(f"Without global: {size_without/1024:.1f} KB (current production)")
    print(f"With global: {size_with/1024:.1f} KB (fixed version)")
    
    return base_model_params, agg_with_global, agg_without_global

if __name__ == "__main__":
    simulate_real_federated_round()
