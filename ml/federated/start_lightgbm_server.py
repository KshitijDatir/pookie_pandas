#!/usr/bin/env python3
"""
LightGBM Server Launcher

Usage: python start_lightgbm_server.py
"""

import sys
import os

# Add project directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'config'))

from lightgbm_federated_server import main

if __name__ == "__main__":
    print("[TREE] Launching LightGBM Federated Learning Server")
    main()
