#!/usr/bin/env python3
"""
Quick Test Launcher for Federated Learning System

This script provides a convenient way to test the federated learning system
with simplified components and clear logging.
"""

import os
import sys
import time
import subprocess
import threading
import signal
from typing import List

def run_command(cmd: List[str], name: str, wait_time: int = 2):
    """Run a command with logging."""
    print(f"🚀 Starting {name}...")
    print(f"   Command: {' '.join(cmd)}")
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Print output in real time
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(f"[{name}] {output.strip()}")
        
        rc = process.poll()
        print(f"🔚 {name} finished with return code {rc}")
        return rc
        
    except Exception as e:
        print(f"❌ {name} failed: {e}")
        return -1

def test_data_addition():
    """Test adding data to MongoDB."""
    print("=" * 60)
    print("🧪 STEP 1: Add Test Transactions")
    print("=" * 60)
    
    # Add enough transactions to trigger FL
    cmd = [sys.executable, "add_test_transaction.py", "--bank", "SBI", "--count", "3"]
    run_command(cmd, "Add Transactions")

def test_server():
    """Test the federated server."""
    print("=" * 60)
    print("🧪 STEP 2: Start Federated Server")
    print("=" * 60)
    
    cmd = [sys.executable, "lightgbm_federated_server.py"]
    
    # Run server in background
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        print(f"🚀 Server started with PID: {process.pid}")
        
        # Let server initialize
        time.sleep(3)
        
        # Check if still running
        if process.poll() is None:
            print("✅ Server is running")
            return process
        else:
            print("❌ Server exited early")
            return None
            
    except Exception as e:
        print(f"❌ Server failed to start: {e}")
        return None

def test_client():
    """Test the simplified client."""
    print("=" * 60)
    print("🧪 STEP 3: Start Test Client")
    print("=" * 60)
    
    cmd = [sys.executable, "test_lightgbm_client.py", "SBI"]
    return run_command(cmd, "Test Client")

def test_full_system():
    """Test the full federated learning system."""
    print("🧪 FEDERATED LEARNING SYSTEM TEST")
    print("=" * 60)
    
    server_process = None
    
    try:
        # Step 1: Add test data
        test_data_addition()
        time.sleep(2)
        
        # Step 2: Start server
        server_process = test_server()
        if not server_process:
            print("❌ Cannot continue without server")
            return
        
        # Step 3: Start client
        time.sleep(2)
        test_client()
        
        print("=" * 60)
        print("✅ TEST COMPLETED")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        
    finally:
        # Clean up server
        if server_process and server_process.poll() is None:
            print("🧹 Cleaning up server process...")
            server_process.terminate()
            time.sleep(2)
            if server_process.poll() is None:
                server_process.kill()

def show_help():
    """Show help information."""
    print("""
🧪 Federated Learning System Test Script

Usage:
    python test_federated_system.py [command]

Commands:
    full      - Run full system test (default)
    data      - Add test transactions only
    server    - Start server only
    client    - Start client only
    help      - Show this help

Examples:
    python test_federated_system.py           # Full test
    python test_federated_system.py data      # Add test data
    python test_federated_system.py server    # Server only
""")

def main():
    """Main function."""
    command = sys.argv[1] if len(sys.argv) > 1 else "full"
    
    if command == "help":
        show_help()
    elif command == "data":
        test_data_addition()
    elif command == "server":
        server_process = test_server()
        if server_process:
            try:
                print("Server running... Press Ctrl+C to stop")
                server_process.wait()
            except KeyboardInterrupt:
                print("🛑 Stopping server...")
                server_process.terminate()
    elif command == "client":
        test_client()
    elif command == "full":
        test_full_system()
    else:
        print(f"❌ Unknown command: {command}")
        show_help()

if __name__ == "__main__":
    main()
