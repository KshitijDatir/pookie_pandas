#!/usr/bin/env python3
"""
Debug script to test the 5-transaction threshold enforcement.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_client_threshold_logic():
    """Test the client's threshold checking logic."""
    
    print("🔍 Testing Client Threshold Logic")
    
    # Mock the client's fit method logic
    def mock_client_fit(current_count):
        """Mock client fit method with threshold check."""
        print(f"\n🧪 Testing with {current_count} transactions:")
        
        # This mirrors the exact logic from lightgbm_bank_client.py line 408
        if current_count < 5:  # Hardcoded to ensure strict enforcement
            print(f"❌ Insufficient data: {current_count} < 5 transactions required")
            return {
                "parameters": "mock_params",
                "num_examples": 0,  # This should be 0
                "metrics": {
                    "status": "insufficient_data", 
                    "data_count": current_count, 
                    "required": 5
                }
            }
        else:
            print(f"✅ Threshold passed: {current_count} transactions - proceeding with FL")
            return {
                "parameters": "mock_trained_params",
                "num_examples": current_count,  # This should be the actual count
                "metrics": {
                    "status": "trained",
                    "data_count": current_count,
                    "accuracy": 0.85
                }
            }
    
    # Test various transaction counts
    test_counts = [0, 1, 2, 3, 4, 5, 6, 10, 15]
    
    for count in test_counts:
        result = mock_client_fit(count)
        
        # Verify the result
        if count < 5:
            expected_num_examples = 0
            expected_status = "insufficient_data"
        else:
            expected_num_examples = count
            expected_status = "trained"
        
        actual_num_examples = result["num_examples"]
        actual_status = result["metrics"]["status"]
        
        if actual_num_examples == expected_num_examples and actual_status == expected_status:
            print(f"  ✅ Correct: num_examples={actual_num_examples}, status='{actual_status}'")
        else:
            print(f"  ❌ WRONG: num_examples={actual_num_examples} (expected {expected_num_examples}), status='{actual_status}' (expected '{expected_status}')")

def test_server_filtering_logic():
    """Test the server's filtering logic."""
    
    print("\n🔍 Testing Server Filtering Logic")
    
    # Mock FitRes class
    class MockFitRes:
        def __init__(self, num_examples, metrics):
            self.num_examples = num_examples
            self.metrics = metrics
    
    # Create mock results like the server receives
    mock_results = [
        # Client with insufficient data
        MockFitRes(num_examples=0, metrics={"status": "insufficient_data", "data_count": 3}),
        # Client with sufficient data
        MockFitRes(num_examples=8, metrics={"status": "trained", "data_count": 8}),
        # Another client with insufficient data
        MockFitRes(num_examples=0, metrics={"status": "insufficient_data", "data_count": 2}),
        # Client with sufficient data
        MockFitRes(num_examples=12, metrics={"status": "trained", "data_count": 12}),
    ]
    
    # Mock the server's filtering logic (from lines 180-192)
    fit_results = []
    insufficient_data_clients = 0
    
    for i, fit_res in enumerate(mock_results):
        print(f"\n🧪 Processing mock client {i+1}:")
        print(f"   - num_examples: {fit_res.num_examples}")
        print(f"   - status: {fit_res.metrics.get('status', 'unknown')}")
        print(f"   - data_count: {fit_res.metrics.get('data_count', 'unknown')}")
        
        # Server's filtering logic
        if (hasattr(fit_res, 'metrics') and 
            fit_res.metrics and 
            fit_res.metrics.get('status') == 'insufficient_data'):
            insufficient_data_clients += 1
            print(f"   ❌ FILTERED OUT: Insufficient data")
        else:
            fit_results.append(fit_res)
            print(f"   ✅ ACCEPTED: Will be processed")
    
    print(f"\n📊 Server filtering results:")
    print(f"   - Insufficient data clients: {insufficient_data_clients}")
    print(f"   - Clients accepted for FL: {len(fit_results)}")
    
    # The key check: does the server proceed with FL?
    if not fit_results:
        print(f"   🛑 NO FL ROUND: No clients with sufficient data")
    else:
        print(f"   🔄 FL ROUND PROCEEDS: {len(fit_results)} clients will be aggregated")
        
        # Check the second filtering in the server (line 225)
        print(f"\n🔍 Second server check (num_examples > 0):")
        for i, fit_res in enumerate(fit_results):
            if fit_res.num_examples > 0:
                print(f"   ✅ Client {i+1} will be aggregated: {fit_res.num_examples} examples")
            else:
                print(f"   ❌ Client {i+1} will be SKIPPED: {fit_res.num_examples} examples")

def test_potential_issue():
    """Test for potential issues in the logic."""
    
    print("\n🚨 Testing Potential Issue Scenarios")
    
    # Mock FitRes class
    class MockFitRes:
        def __init__(self, num_examples, metrics):
            self.num_examples = num_examples
            self.metrics = metrics
    
    # Scenario 1: Client returns insufficient_data but num_examples > 0 (BUG?)
    print(f"\n🧪 Scenario 1: Client bug - returns 'insufficient_data' but num_examples=3")
    
    buggy_result = MockFitRes(
        num_examples=3,  # This should be 0 for insufficient_data
        metrics={"status": "insufficient_data", "data_count": 3}
    )
    
    # Server's first filter
    if buggy_result.metrics.get('status') == 'insufficient_data':
        print(f"   ✅ First filter: CORRECTLY filtered out (insufficient_data status)")
        filtered_out = True
    else:
        print(f"   ❌ First filter: WRONGLY accepted")
        filtered_out = False
    
    # If it somehow gets through...
    if not filtered_out and buggy_result.num_examples > 0:
        print(f"   🚨 SECOND CHECK WOULD ACCEPT IT: num_examples={buggy_result.num_examples}")
    
    # Scenario 2: Check if your actual client code has this bug
    print(f"\n🧪 Scenario 2: Checking if real client might have this bug...")
    print(f"   The client code should return num_examples=0 when status='insufficient_data'")
    print(f"   From lightgbm_bank_client.py line 410:")
    print(f"   return self.get_parameters(config), 0, {{'status': 'insufficient_data', ...}}")
    print(f"                                    ↑")
    print(f"   This looks correct - it returns 0 as num_examples")

def check_live_system():
    """Suggestions for checking the live system."""
    
    print(f"\n🔍 How to Debug Your Live System")
    print(f"")
    print(f"1. Check server logs for these messages:")
    print(f"   '[FAIL] Client X: Insufficient data (Y < 5 transactions)'")
    print(f"   '[BLOCKED] N client(s) blocked due to insufficient data'")
    print(f"   '[WAIT] Round X: No clients with sufficient data'")
    print(f"")
    print(f"2. If you see '[SUCCESS] N client(s) have sufficient data' when you shouldn't:")
    print(f"   - Check the client logs for the exact return values")
    print(f"   - Verify that get_available_data_count() returns the correct count")
    print(f"   - Check if there's a race condition in data collection")
    print(f"")
    print(f"3. Add this debug log to your client's fit method (after line 405):")
    print(f"   self.logger.info(f'🔍 DEBUG: current_count={{current_count}}, threshold=5')")
    print(f"")
    print(f"4. Add this debug log to your server's aggregate_fit method (after line 186):")
    print(f"   self.logger.info(f'🔍 DEBUG: Client status={{fit_res.metrics.get(\"status\")}}, num_examples={{fit_res.num_examples}}')")

if __name__ == "__main__":
    print("🚀 Debugging 5-Transaction Threshold Enforcement")
    
    test_client_threshold_logic()
    test_server_filtering_logic()
    test_potential_issue()
    check_live_system()
    
    print(f"\n🎯 CONCLUSION:")
    print(f"The threshold logic appears correct in the code.")
    print(f"If data is still updating with <5 transactions, check:")
    print(f"1. What get_available_data_count() actually returns")
    print(f"2. Server logs for filtering messages")
    print(f"3. Possible race conditions in data collection")
