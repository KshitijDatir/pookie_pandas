#!/usr/bin/env python3
"""
Quick Model Version Status Checker
"""

from model_version_manager import ModelVersionManager

def main():
    print("🔍 Quick Model Status Check")
    print("=" * 50)
    
    manager = ModelVersionManager()
    
    # Show current vs original comparison
    manager.compare_models()
    
    # Show version count
    if manager.metadata.get("versions"):
        total_versions = len(manager.metadata["versions"])
        print(f"\n📦 Total Backup Versions: {total_versions}")
        
        # Show latest version
        latest = manager.metadata["versions"][-1] if manager.metadata["versions"] else None
        if latest:
            print(f"Latest Backup: {latest['version']} ({latest['model_info'].get('file_size_kb', 'Unknown')} KB)")
    else:
        print("\n📦 No backup versions found")
    
    print("=" * 50)
    print("💡 Use 'python model_version_manager.py' for full management")

if __name__ == "__main__":
    main()
