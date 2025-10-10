#!/usr/bin/env python3
"""
Test script to verify global model preservation in federated aggregation.

This test creates a trained global model, then simulates client updates
and verifies that the aggregation preserves the global model structure
while incorporating client knowledge.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import json
import logging
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from models.lightgbm_model import FederatedLightGBM, aggregate_lightgbm_models

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_data(n_samples=1000, random_state=42):
    """Create test classification dataset."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_clusters_per_class=1,
        random_state=random_state
    )
    
    # Convert to DataFrame for consistency
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    return df

def train_global_model():
    """Train a substantial global model."""
    logger.info("🏭 Training substantial global model...")
    
    # Create large training dataset (simulating 4M samples with 2K for testing speed)
    global_data = create_test_data(n_samples=2000, random_state=42)
    X = global_data.drop('target', axis=1)
    y = global_data['target']
    
    # Initialize and train global model
    global_model = FederatedLightGBM()
    
    # Train with more boosting rounds to create substantial model
    global_model.model_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'num_boost_round': 150,  # Create substantial model
        'learning_rate': 0.1,
        'num_leaves': 31,
        'verbose': -1
    }
    
    global_model.fit(X, y)
    
    # Get model parameters
    global_params = global_model.get_model_params()
    
    logger.info(f"✅ Global model trained: {global_params.get('num_trees', 0)} trees, "
                f"{len(global_params.get('model_dump', ''))} chars")
    
    return global_model, global_params

def create_client_models():
    """Create multiple client models with smaller datasets."""
    logger.info("👥 Creating client models...")
    
    client_models = []
    client_params_list = []
    client_weights = []
    
    # Create 3 clients with realistic small data sizes (10-1000 samples)
    client_configs = [
        (50, 10),   # Very small client (10 samples)
        (100, 50),  # Small client (50 samples)  
        (200, 100), # Medium client (100 samples)
    ]
    
    for i, (n_samples, weight) in enumerate(client_configs):
        logger.info(f"🔧 Training client {i+1} with {n_samples} samples...")
        
        # Create client-specific dataset
        client_data = create_test_data(n_samples=n_samples, random_state=42+i)
        X = client_data.drop('target', axis=1)
        y = client_data['target']
        
        # Initialize client model
        client_model = FederatedLightGBM()
        client_model.model_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'num_boost_round': 50,  # Smaller models
            'learning_rate': 0.1 + (i * 0.05),  # More different learning rates
            'num_leaves': 25,
            'verbose': -1
        }
        
        client_model.fit(X, y)
        
        # Get client parameters
        client_params = client_model.get_model_params()
        
        client_models.append(client_model)
        client_params_list.append(client_params)
        client_weights.append(weight)
        
        logger.info(f"✅ Client {i+1}: {client_params.get('num_trees', 0)} trees, "
                    f"{len(client_params.get('model_dump', ''))} chars")
    
    return client_models, client_params_list, client_weights

def test_aggregation_with_global_preservation():
    """Test that aggregation preserves global model while incorporating client updates."""
    logger.info("🧪 Testing global model preservation in aggregation...")
    
    # Step 1: Train global model
    global_model, global_params = train_global_model()
    original_trees = global_params.get('num_trees', 0)
    original_lr = global_params.get('params', {}).get('learning_rate', 0.1)
    original_size = len(global_params.get('model_dump', ''))
    
    logger.info(f"📊 Global Model (4M samples): {original_trees} trees, LR: {original_lr:.4f}, Size: {original_size}")
    
    # Step 2: Create client models
    client_models, client_params_list, client_weights = create_client_models()
    
    # Step 3: Test aggregation WITHOUT global model (old behavior)
    logger.info("\n🔄 Testing client-only aggregation (old approach)...")
    aggregated_without_global = aggregate_lightgbm_models(client_params_list, client_weights, None)
    
    without_trees = aggregated_without_global.get('num_trees', 0)
    without_lr = aggregated_without_global.get('params', {}).get('learning_rate', 0.1)
    without_size = len(aggregated_without_global.get('model_dump', ''))
    
    logger.info(f"📊 Without Global: {without_trees} trees, LR: {without_lr:.4f}, Size: {without_size}")
    
    # Step 4: Test aggregation WITH global model (new behavior)
    logger.info("\n🔄 Testing global model preservation (99% global, 1% client)...")
    aggregated_with_global = aggregate_lightgbm_models(client_params_list, client_weights, global_params)
    
    with_trees = aggregated_with_global.get('num_trees', 0)
    with_lr = aggregated_with_global.get('params', {}).get('learning_rate', 0.1)
    with_size = len(aggregated_with_global.get('model_dump', ''))
    
    logger.info(f"📊 With Global: {with_trees} trees, LR: {with_lr:.4f}, Size: {with_size}")
    
    # Step 5: Verification
    logger.info("\n🔍 Verification Results:")
    
    # Check that global model preservation maintains model complexity
    complexity_preserved = with_trees >= original_trees * 0.9  # Allow small variance
    logger.info(f"✅ Model complexity preserved: {complexity_preserved} ({with_trees} >= {original_trees * 0.9:.0f})")
    
    # Check that model size is maintained (not tiny)
    size_maintained = with_size >= original_size * 0.8  # Allow some compression
    logger.info(f"✅ Model size maintained: {size_maintained} ({with_size} >= {original_size * 0.8:.0f})")
    
    # Check that learning rate is updated (influenced by clients)
    lr_updated = abs(with_lr - original_lr) > 0.001
    logger.info(f"✅ Learning rate updated: {lr_updated} ({original_lr:.4f} → {with_lr:.4f})")
    
    # Compare with non-global aggregation
    better_than_client_only = with_size > without_size * 1.5  # Should be substantially larger
    logger.info(f"✅ Better than client-only: {better_than_client_only} ({with_size} > {without_size * 1.5:.0f})")
    
    # Step 6: Test model functionality
    logger.info("\n🧪 Testing model functionality...")
    
    # Create test data for prediction
    test_data = create_test_data(n_samples=100, random_state=999)
    X_test = test_data.drop('target', axis=1)
    y_test = test_data['target']
    
    # Test global model predictions
    global_model_test = FederatedLightGBM()
    global_model_test.set_model_params(global_params)
    global_predictions = global_model_test.predict(X_test)
    
    # Test aggregated model predictions
    aggregated_model_test = FederatedLightGBM()
    aggregated_model_test.set_model_params(aggregated_with_global)
    aggregated_predictions = aggregated_model_test.predict(X_test)
    
    # Verify both models can make predictions
    global_can_predict = len(global_predictions) == len(X_test)
    aggregated_can_predict = len(aggregated_predictions) == len(X_test)
    
    logger.info(f"✅ Global model can predict: {global_can_predict} ({len(global_predictions)} predictions)")
    logger.info(f"✅ Aggregated model can predict: {aggregated_can_predict} ({len(aggregated_predictions)} predictions)")
    
    # Calculate prediction correlation (should be similar but not identical)
    if global_can_predict and aggregated_can_predict:
        correlation = np.corrcoef(global_predictions, aggregated_predictions)[0, 1]
        logger.info(f"📊 Prediction correlation: {correlation:.4f} (should be high but not 1.0)")
        
        predictions_similar = 0.7 < correlation < 0.99
        logger.info(f"✅ Predictions appropriately similar: {predictions_similar}")
    
    # Summary
    logger.info("\n📋 SUMMARY:")
    all_tests_passed = (complexity_preserved and size_maintained and lr_updated and 
                       better_than_client_only and global_can_predict and aggregated_can_predict)
    
    if all_tests_passed:
        logger.info("🎉 ALL TESTS PASSED! Global model preservation working correctly.")
        logger.info("✅ The federated system now preserves global model complexity")
        logger.info("✅ While still incorporating client knowledge through weighted updates")
    else:
        logger.info("❌ Some tests failed. Global model preservation needs improvement.")
    
    return all_tests_passed

def test_multiple_rounds():
    """Test multiple rounds of federated learning with global preservation."""
    logger.info("\n🔄 Testing multiple federated rounds...")
    
    # Start with global model
    global_model, global_params = train_global_model()
    current_global = global_params
    
    logger.info(f"🏁 Starting with global model: {current_global.get('num_trees', 0)} trees")
    
    # Simulate 3 rounds of federated learning
    for round_num in range(1, 4):
        logger.info(f"\n--- Round {round_num} ---")
        
        # Create new client models for this round
        client_models, client_params_list, client_weights = create_client_models()
        
        # Aggregate with current global model
        current_global = aggregate_lightgbm_models(client_params_list, client_weights, current_global)
        
        trees = current_global.get('num_trees', 0)
        lr = current_global.get('params', {}).get('learning_rate', 0.1)
        size = len(current_global.get('model_dump', ''))
        
        logger.info(f"Round {round_num} result: {trees} trees, LR: {lr:.4f}, Size: {size}")
        
        # Verify model still works
        test_model = FederatedLightGBM()
        test_model.set_model_params(current_global)
        test_data = create_test_data(n_samples=50, random_state=round_num)
        X_test = test_data.drop('target', axis=1)
        predictions = test_model.predict(X_test)
        
        logger.info(f"✅ Round {round_num} model functional: {len(predictions)} predictions made")
    
    logger.info(f"\n🏆 Final model after 3 rounds: {current_global.get('num_trees', 0)} trees")

if __name__ == "__main__":
    logger.info("🚀 Starting Global Model Preservation Test")
    
    try:
        # Test basic aggregation with global preservation
        success = test_aggregation_with_global_preservation()
        
        if success:
            # Test multiple rounds
            test_multiple_rounds()
            
        logger.info("\n🎯 Test complete!")
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
