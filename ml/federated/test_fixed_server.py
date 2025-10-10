#!/usr/bin/env python3
"""
Test the fixed server initialization to see if it properly loads the 4M sample base model.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pickle
import logging
from lightgbm_federated_server import LightGBMFederatedStrategy

def test_server_initialization():
    """Test the server initialization with the actual base model."""
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("TEST")
    
    print("🚀 Testing Fixed Server Initialization")
    
    # Check if base model exists
    base_model_path = "trained_models/lightgbm_model.pkl"
    if not os.path.exists(base_model_path):
        print(f"❌ Base model not found at {base_model_path}")
        print("Creating a test base model...")
        
        # Create base model for testing
        from test_real_federated_flow import create_realistic_base_model, save_base_model
        base_model_obj, base_model_params = create_realistic_base_model()
        save_base_model(base_model_obj, base_model_params, base_model_path)
    
    # Test server initialization
    print(f"\n🔧 Initializing LightGBM server with {base_model_path}...")
    
    try:
        # Create strategy (this will test base model loading)
        strategy = LightGBMFederatedStrategy(base_model_path=base_model_path)
        
        print(f"✅ Strategy created successfully")
        print(f"📊 Base model fitted: {strategy.base_model.is_fitted if strategy.base_model else 'None'}")
        
        if strategy.base_model and strategy.base_model.is_fitted:
            params = strategy.base_model.get_model_params()
            if params:
                print(f"📊 Base model trees: {params.get('num_trees', 'unknown')}")
                print(f"📊 Base model samples: {params.get('total_samples', 'unknown'):,}")
                print(f"📊 Model dump size: {len(params.get('model_dump', ''))}")
            
        # Test parameter initialization
        print(f"\n🔧 Testing parameter initialization...")
        initial_params = strategy.initialize_parameters(None)
        
        if initial_params:
            print(f"✅ Parameters initialized successfully")
            
            # Test decoding the parameters
            import flwr as fl
            import numpy as np
            import json
            
            try:
                param_arrays = fl.common.parameters_to_ndarrays(initial_params)
                if len(param_arrays) > 0:
                    # Decode back to check content
                    byte_array = param_arrays[0].astype(np.uint8)
                    model_bytes = byte_array.tobytes()
                    json_str = model_bytes.decode('utf-8', errors='ignore')
                    
                    if len(json_str) > 100:  # Valid content
                        model_data = json.loads(json_str)
                        print(f"📊 Decoded parameters:")
                        print(f"   - Type: {model_data.get('model_type', 'unknown')}")
                        print(f"   - Trees: {model_data.get('num_trees', 'unknown')}")  
                        print(f"   - Samples: {model_data.get('total_samples', 'unknown'):,}")
                        print(f"   - Model dump size: {len(model_data.get('model_dump', ''))}")
                        
                        # This is the key test - does it have the 4M samples?
                        total_samples = model_data.get('total_samples', 0)
                        if total_samples >= 1000000:
                            print(f"🎉 SUCCESS: Large model detected ({total_samples:,} samples)")
                        else:
                            print(f"❌ FAILURE: Small model detected ({total_samples:,} samples)")
                    else:
                        print(f"❌ Decoded JSON too short: {len(json_str)} chars")
                        
            except Exception as decode_error:
                print(f"❌ Failed to decode parameters: {decode_error}")
        else:
            print(f"❌ Parameter initialization failed")
            
    except Exception as e:
        print(f"❌ Server initialization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_server_initialization()
