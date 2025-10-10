#!/usr/bin/env python3
"""
Model Version Management System for LightGBM Federated Learning
Handles model versioning, backup, and restoration
"""

import os
import shutil
import pickle
import json
from datetime import datetime
from pathlib import Path
import lightgbm as lgb

class ModelVersionManager:
    def __init__(self, models_dir="trained_models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Version history directory
        self.versions_dir = self.models_dir / "versions"
        self.versions_dir.mkdir(exist_ok=True)
        
        # Metadata file
        self.metadata_file = self.models_dir / "model_metadata.json"
        
        # Current model paths
        self.current_model_path = self.models_dir / "latest_lightgbm_federated.pkl"
        self.original_model_path = self.models_dir / "lightgbm_model.pkl"
        
        # Load or initialize metadata
        self.metadata = self._load_metadata()
    
    def _load_metadata(self):
        """Load model metadata or create if doesn't exist"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        else:
            return {
                "versions": [],
                "current_version": None,
                "original_backup": None
            }
    
    def _save_metadata(self):
        """Save model metadata"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def get_model_info(self, model_path):
        """Get information about a model file"""
        if not Path(model_path).exists():
            return None
            
        stats = os.stat(model_path)
        
        # Try to load model to get more info
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            info = {
                "file_size_bytes": stats.st_size,
                "file_size_kb": round(stats.st_size / 1024, 2),
                "last_modified": datetime.fromtimestamp(stats.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                "model_type": type(model).__name__,
            }
            
            # Try to get LightGBM specific info
            if hasattr(model, 'num_trees'):
                info["num_trees"] = model.num_trees()
            if hasattr(model, 'num_feature'):
                info["num_features"] = model.num_feature()
                
            return info
            
        except Exception as e:
            return {
                "file_size_bytes": stats.st_size,
                "file_size_kb": round(stats.st_size / 1024, 2),
                "last_modified": datetime.fromtimestamp(stats.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                "error": str(e)
            }
    
    def backup_current_model(self, description="Manual backup"):
        """Create a versioned backup of the current model"""
        if not self.current_model_path.exists():
            print(f"❌ No current model found at {self.current_model_path}")
            return False
        
        # Generate version info
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_name = f"v_{timestamp}"
        backup_path = self.versions_dir / f"{version_name}_lightgbm_federated.pkl"
        
        # Copy model file
        shutil.copy2(self.current_model_path, backup_path)
        
        # Get model info
        model_info = self.get_model_info(self.current_model_path)
        
        # Add to metadata
        version_info = {
            "version": version_name,
            "timestamp": timestamp,
            "description": description,
            "file_path": str(backup_path),
            "model_info": model_info
        }
        
        self.metadata["versions"].append(version_info)
        self._save_metadata()
        
        print(f"✅ Model backed up as version {version_name}")
        print(f"   Size: {model_info.get('file_size_kb', 'Unknown')} KB")
        print(f"   Path: {backup_path}")
        
        return version_name
    
    def restore_version(self, version_name):
        """Restore a specific version as the current model"""
        # Find the version
        version_info = None
        for v in self.metadata["versions"]:
            if v["version"] == version_name:
                version_info = v
                break
        
        if not version_info:
            print(f"❌ Version {version_name} not found")
            return False
        
        backup_path = Path(version_info["file_path"])
        if not backup_path.exists():
            print(f"❌ Backup file not found: {backup_path}")
            return False
        
        # Backup current model first
        if self.current_model_path.exists():
            self.backup_current_model("Auto backup before restoration")
        
        # Restore the version
        shutil.copy2(backup_path, self.current_model_path)
        
        self.metadata["current_version"] = version_name
        self._save_metadata()
        
        print(f"✅ Restored version {version_name} as current model")
        model_info = self.get_model_info(self.current_model_path)
        print(f"   Size: {model_info.get('file_size_kb', 'Unknown')} KB")
        
        return True
    
    def restore_original_model(self):
        """Restore the original large model"""
        if not self.original_model_path.exists():
            print(f"❌ Original model not found at {self.original_model_path}")
            return False
        
        # Backup current model first
        if self.current_model_path.exists():
            self.backup_current_model("Auto backup before original restoration")
        
        # Copy original to current
        shutil.copy2(self.original_model_path, self.current_model_path)
        
        self.metadata["current_version"] = "original_restored"
        self._save_metadata()
        
        print("✅ Original model restored as current model")
        model_info = self.get_model_info(self.current_model_path)
        print(f"   Size: {model_info.get('file_size_kb', 'Unknown')} KB")
        
        return True
    
    def list_versions(self):
        """List all available versions"""
        print("\n📋 Model Version History:")
        print("=" * 80)
        
        # Show current model
        if self.current_model_path.exists():
            current_info = self.get_model_info(self.current_model_path)
            print(f"🔹 CURRENT MODEL:")
            print(f"   Path: {self.current_model_path}")
            print(f"   Size: {current_info.get('file_size_kb', 'Unknown')} KB")
            print(f"   Modified: {current_info.get('last_modified', 'Unknown')}")
            if 'num_trees' in current_info:
                print(f"   Trees: {current_info['num_trees']}")
        
        # Show original model
        if self.original_model_path.exists():
            original_info = self.get_model_info(self.original_model_path)
            print(f"\n🔹 ORIGINAL MODEL:")
            print(f"   Path: {self.original_model_path}")
            print(f"   Size: {original_info.get('file_size_kb', 'Unknown')} KB")
            print(f"   Modified: {original_info.get('last_modified', 'Unknown')}")
            if 'num_trees' in original_info:
                print(f"   Trees: {original_info['num_trees']}")
        
        # Show version history
        if self.metadata["versions"]:
            print(f"\n🔹 BACKED UP VERSIONS ({len(self.metadata['versions'])}):")
            for i, version in enumerate(reversed(self.metadata["versions"]), 1):
                print(f"   {i}. {version['version']}")
                print(f"      Description: {version['description']}")
                print(f"      Size: {version['model_info'].get('file_size_kb', 'Unknown')} KB")
                print(f"      Timestamp: {version['timestamp']}")
                if i < len(self.metadata["versions"]):
                    print()
        else:
            print("\n🔹 No backed up versions found")
        
        print("=" * 80)
    
    def compare_models(self):
        """Compare current model with original"""
        print("\n🔍 Model Comparison:")
        print("=" * 60)
        
        current_info = self.get_model_info(self.current_model_path) if self.current_model_path.exists() else None
        original_info = self.get_model_info(self.original_model_path) if self.original_model_path.exists() else None
        
        if current_info and original_info:
            print(f"Current Model: {current_info.get('file_size_kb', 'Unknown')} KB")
            print(f"Original Model: {original_info.get('file_size_kb', 'Unknown')} KB")
            
            size_diff = original_info.get('file_size_kb', 0) - current_info.get('file_size_kb', 0)
            print(f"Size Difference: {size_diff:.2f} KB")
            
            if size_diff > 100:  # If original is significantly larger
                print("⚠️  Current model is much smaller than original!")
                print("   This suggests the current model was trained with limited data.")
                print("   Consider restoring the original model if needed.")
        
        print("=" * 60)

def main():
    """Interactive model version management"""
    manager = ModelVersionManager()
    
    while True:
        print("\n🎯 Model Version Manager")
        print("1. List all versions")
        print("2. Compare current vs original")
        print("3. Backup current model")
        print("4. Restore original model")
        print("5. Restore specific version")
        print("6. Exit")
        
        choice = input("\nSelect option (1-6): ").strip()
        
        if choice == '1':
            manager.list_versions()
        
        elif choice == '2':
            manager.compare_models()
        
        elif choice == '3':
            desc = input("Enter backup description (optional): ").strip() or "Manual backup"
            manager.backup_current_model(desc)
        
        elif choice == '4':
            confirm = input("Restore original model? This will backup current model first (y/N): ").strip().lower()
            if confirm == 'y':
                manager.restore_original_model()
        
        elif choice == '5':
            manager.list_versions()
            version = input("\nEnter version name to restore: ").strip()
            if version:
                manager.restore_version(version)
        
        elif choice == '6':
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please select 1-6.")

if __name__ == "__main__":
    main()
