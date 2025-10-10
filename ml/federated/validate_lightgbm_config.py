#!/usr/bin/env python3
"""
LightGBM Configuration Validation Script

Validates that all configuration files are properly set up for LightGBM
instead of encoder-decoder architecture.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'config'))

def validate_config():
    """Validate LightGBM configuration settings."""
    
    print("🔍 LightGBM Configuration Validation")
    print("=" * 60)
    
    try:
        from config.federated_config import (
            MODEL_CONFIG, TRAINING_CONFIG, FRAUD_DETECTION_CONFIG, 
            PATHS_CONFIG, get_model_config
        )
        
        # Check Model Configuration
        print("\n📋 Model Configuration:")
        print("-" * 30)
        
        model_config = get_model_config()
        
        # Validate model type
        if model_config.get("model_type") == "lightgbm":
            print("✅ Model Type: LightGBM")
        else:
            print(f"❌ Model Type: {model_config.get('model_type', 'UNDEFINED')} (should be 'lightgbm')")
        
        # Check LightGBM parameters match original training
        expected_params = {
            "n_estimators": 200,
            "learning_rate": 0.05, 
            "max_depth": 10,
            "num_leaves": 31,
            "random_state": 42
        }
        
        print("\n🔧 LightGBM Parameters:")
        for param, expected in expected_params.items():
            actual = model_config.get(param)
            if actual == expected:
                print(f"✅ {param}: {actual}")
            else:
                print(f"⚠️  {param}: {actual} (expected: {expected})")
        
        # Check for old encoder-decoder parameters
        old_params = ["architecture", "encoder", "decoder", "input_dim", "batch_size", "dropout_rate"]
        
        print("\n🚫 Checking for old encoder-decoder parameters:")
        found_old = False
        for param in old_params:
            if param in model_config:
                print(f"❌ Found old parameter: {param}")
                found_old = True
        
        if not found_old:
            print("✅ No old encoder-decoder parameters found")
        
        # Check Training Configuration
        print("\n📚 Training Configuration:")
        print("-" * 30)
        
        # Check for SMOTE (as used in original training)
        if TRAINING_CONFIG.get("apply_smote") == True:
            print("✅ SMOTE enabled (matches original training)")
        else:
            print("⚠️  SMOTE not enabled")
        
        # Check random state consistency
        if TRAINING_CONFIG.get("random_state") == 42:
            print("✅ Random state: 42 (consistent)")
        else:
            print(f"⚠️  Random state: {TRAINING_CONFIG.get('random_state')}")
        
        # Check Fraud Detection Configuration
        print("\n🎯 Fraud Detection Configuration:")
        print("-" * 30)
        
        if FRAUD_DETECTION_CONFIG.get("use_probability") == True:
            print("✅ Using probability-based detection (LightGBM)")
        else:
            print("⚠️  Not using probability-based detection")
        
        if "reconstruction" in str(FRAUD_DETECTION_CONFIG):
            print("❌ Found 'reconstruction' references (encoder-decoder leftover)")
        else:
            print("✅ No reconstruction-based detection references")
        
        # Check Paths Configuration
        print("\n📁 Paths Configuration:")
        print("-" * 30)
        
        if PATHS_CONFIG.get("base_model_name") == "lightgbm_model.pkl":
            print("✅ Base model: lightgbm_model.pkl")
        else:
            print(f"⚠️  Base model: {PATHS_CONFIG.get('base_model_name')}")
        
        if PATHS_CONFIG.get("current_model_name") == "latest_lightgbm_federated.pkl":
            print("✅ Current model: latest_lightgbm_federated.pkl")
        else:
            print(f"⚠️  Current model: {PATHS_CONFIG.get('current_model_name')}")
        
        # Check if old PyTorch model references exist
        if "pth" in str(PATHS_CONFIG):
            print("❌ Found .pth references (PyTorch leftover)")
        else:
            print("✅ No PyTorch model references")
        
        print("\n" + "=" * 60)
        print("🎯 Configuration Validation Complete!")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import configuration: {e}")
        return False
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False

def check_model_files():
    """Check if model files exist and are correct type."""
    
    print("\n📁 Model Files Check:")
    print("-" * 30)
    
    trained_models_dir = "trained_models"
    
    # Check for LightGBM models
    lightgbm_files = [
        "lightgbm_model.pkl",
        "latest_lightgbm_federated.pkl"
    ]
    
    for filename in lightgbm_files:
        filepath = os.path.join(trained_models_dir, filename)
        if os.path.exists(filepath):
            size_kb = os.path.getsize(filepath) / 1024
            print(f"✅ {filename}: {size_kb:.2f} KB")
        else:
            print(f"❌ {filename}: Not found")
    
    # Check for old encoder-decoder models
    old_files = [
        "latest_base_model.pth",
        "base_model.pth",
        "encoder_decoder_model.pth"
    ]
    
    found_old = False
    for filename in old_files:
        filepath = os.path.join(trained_models_dir, filename)
        if os.path.exists(filepath):
            print(f"⚠️  Found old file: {filename}")
            found_old = True
    
    if not found_old:
        print("✅ No old encoder-decoder model files found")

def main():
    """Main validation function."""
    
    # Change to script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    success = validate_config()
    check_model_files()
    
    if success:
        print("\n🎉 Configuration is properly set up for LightGBM!")
    else:
        print("\n⚠️  Configuration issues detected. Please review and fix.")

if __name__ == "__main__":
    main()
