#!/usr/bin/env python3
"""
LightGBM Client Launcher

Usage: python start_lightgbm_client.py [BANK_ID]
Example: python start_lightgbm_client.py SBI
"""

import sys
import os

# Add project directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'config'))

from lightgbm_bank_client import start_lightgbm_bank_client
from config.federated_config import get_bank_config, get_server_config

def main():
    """Launch LightGBM bank client."""
    
    # Get bank ID from command line or use default
    bank_id = sys.argv[1] if len(sys.argv) > 1 else "SBI"
    
    print(f"🌳 Starting LightGBM client for: {bank_id}")
    
    # Get configurations
    try:
        bank_config = get_bank_config(bank_id)
        server_config = get_server_config()
        
        # Start LightGBM client
        start_lightgbm_bank_client(
            bank_id=bank_config["bank_id"],
            mongo_config=bank_config["mongo_config"],
            server_address=server_config["server_address"]
        )
        
    except Exception as e:
        print(f"❌ Failed to start LightGBM client: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
