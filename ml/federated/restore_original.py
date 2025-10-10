#!/usr/bin/env python3
"""
Restore the original large model as current model.
"""

from model_version_manager import ModelVersionManager

def main():
    print("🔄 Restoring Original Model")
    print("=" * 40)
    
    manager = ModelVersionManager()
    
    # Show current status
    print("Current Status:")
    manager.compare_models()
    
    # Restore original
    print("\n🔄 Restoring original model...")
    success = manager.restore_original_model()
    
    if success:
        print("\n✅ Original model restored successfully!")
        manager.compare_models()
    else:
        print("\n❌ Failed to restore original model")

if __name__ == "__main__":
    main()
