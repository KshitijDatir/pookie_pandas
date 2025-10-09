#!/usr/bin/env python3
"""Debug LightGBM model dump format"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import sys
import os

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'models'))

def debug_model_dump():
    print("🔍 Debugging LightGBM Model Dump Format")
    print("=" * 50)
    
    # Create sample data
    X = pd.DataFrame(np.random.rand(100, 5), columns=[f'feature_{i}' for i in range(5)])
    y = pd.Series(np.random.randint(0, 2, 100))
    
    # Train LightGBM model
    model = lgb.LGBMClassifier(n_estimators=3, random_state=42, verbose=-1)
    model.fit(X, y)
    
    print(f"Model attributes:")
    for attr in dir(model):
        if not attr.startswith('_') and hasattr(model, attr):
            try:
                val = getattr(model, attr)
                if not callable(val):
                    print(f"  {attr}: {type(val)} = {str(val)[:100]}")
            except:
                pass
    
    print(f"\nModel private attributes:")
    for attr in dir(model):
        if attr.startswith('_') and hasattr(model, attr):
            try:
                val = getattr(model, attr)
                if not callable(val) and not attr.startswith('__'):
                    print(f"  {attr}: {type(val)}")
            except:
                pass
    
    # Try different ways to get model dump
    print(f"\n🌳 Model Dump Attempts:")
    
    try:
        if hasattr(model, 'booster_'):
            dump1 = model.booster_.model_to_string()
            print(f"1. model.booster_.model_to_string(): {len(dump1)} chars")
            print(f"   First 200 chars: {dump1[:200]}")
            print(f"   Starts with 'tree': {dump1.startswith('tree')}")
            print(f"   Contains 'Tree=': {'Tree=' in dump1}")
        else:
            print("1. No booster_ attribute")
    except Exception as e:
        print(f"1. Error: {e}")
    
    try:
        if hasattr(model, '_Booster'):
            dump2 = model._Booster.model_to_string()
            print(f"2. model._Booster.model_to_string(): {len(dump2)} chars")
        else:
            print("2. No _Booster attribute")
    except Exception as e:
        print(f"2. Error: {e}")
    
    # Check if we can create booster from dump
    try:
        if hasattr(model, 'booster_'):
            dump = model.booster_.model_to_string()
            new_booster = lgb.Booster(model_str=dump)
            print(f"3. ✅ Successfully created booster from dump")
            
            # Test prediction
            pred1 = model.predict(X[:5])
            pred2 = new_booster.predict(X[:5].values)
            pred2_class = (pred2 > 0.5).astype(int)
            
            print(f"   Original predictions: {pred1}")
            print(f"   Booster predictions (prob): {pred2}")
            print(f"   Booster predictions (class): {pred2_class}")
            print(f"   Match: {np.array_equal(pred1, pred2_class)}")
            
    except Exception as e:
        print(f"3. Error creating booster: {e}")

if __name__ == "__main__":
    debug_model_dump()
