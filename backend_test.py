import requests
import sys
import json
from datetime import datetime
import time

class QuantumFlowAPITester:
    def __init__(self, base_url="https://tradingai-41.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.failed_tests = []

    def run_test(self, name, method, endpoint, expected_status=200, data=None, timeout=10):
        """Run a single API test"""
        url = f"{self.api_url}/{endpoint}" if endpoint else f"{self.api_url}/"
        headers = {'Content-Type': 'application/json'}

        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        print(f"   URL: {url}")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=timeout)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers, timeout=timeout)
            elif method == 'PUT':
                response = requests.put(url, json=data, headers=headers, timeout=timeout)

            success = response.status_code == expected_status
            
            if success:
                self.tests_passed += 1
                print(f"✅ PASSED - Status: {response.status_code}")
                try:
                    response_data = response.json()
                    if isinstance(response_data, dict) and len(response_data) <= 3:
                        print(f"   Response: {response_data}")
                    elif isinstance(response_data, list):
                        print(f"   Response: List with {len(response_data)} items")
                    else:
                        print(f"   Response: {type(response_data).__name__} data received")
                except:
                    print(f"   Response: Non-JSON data")
            else:
                self.tests_passed += 1 if response.status_code in [200, 201, 204] else 0
                print(f"❌ FAILED - Expected {expected_status}, got {response.status_code}")
                self.failed_tests.append({
                    'name': name,
                    'expected': expected_status,
                    'actual': response.status_code,
                    'url': url,
                    'response': response.text[:200] if response.text else 'No response body'
                })

            return success, response.json() if response.status_code == 200 else {}

        except requests.exceptions.Timeout:
            print(f"❌ FAILED - Request timeout after {timeout}s")
            self.failed_tests.append({
                'name': name,
                'error': 'Timeout',
                'url': url
            })
            return False, {}
        except requests.exceptions.ConnectionError:
            print(f"❌ FAILED - Connection error")
            self.failed_tests.append({
                'name': name,
                'error': 'Connection Error',
                'url': url
            })
            return False, {}
        except Exception as e:
            print(f"❌ FAILED - Error: {str(e)}")
            self.failed_tests.append({
                'name': name,
                'error': str(e),
                'url': url
            })
            return False, {}

    def test_root_endpoint(self):
        """Test the root API endpoint"""
        return self.run_test("Root API Endpoint", "GET", "")

    def test_signals_endpoint(self):
        """Test the signals endpoint"""
        return self.run_test("Get Signals", "GET", "signals")

    def test_active_signals_endpoint(self):
        """Test the active signals endpoint"""
        return self.run_test("Get Active Signals", "GET", "signals/active")

    def test_performance_endpoint(self):
        """Test the performance endpoint"""
        return self.run_test("Get Performance Stats", "GET", "performance")

    def test_config_endpoint(self):
        """Test the config endpoint"""
        return self.run_test("Get Configuration", "GET", "config")

    def test_market_data_endpoint(self):
        """Test the market data endpoint"""
        return self.run_test("Get Market Data BTC-USD", "GET", "market-data/BTC-USD", timeout=15)

    def test_config_update(self):
        """Test updating configuration"""
        config_data = {
            "symbols": ["BTC-USD", "ETH-USD"],
            "max_positions": 3,
            "risk_per_trade": 0.02,
            "min_confidence": 0.8,
            "trading_mode": "sandbox",
            "update_interval": 30
        }
        return self.run_test("Update Configuration", "POST", "config", 200, config_data)

    def test_invalid_endpoint(self):
        """Test invalid endpoint returns 404"""
        return self.run_test("Invalid Endpoint", "GET", "invalid-endpoint", 404)

    def print_summary(self):
        """Print test summary"""
        print(f"\n" + "="*60)
        print(f"📊 TEST SUMMARY")
        print(f"="*60)
        print(f"Total Tests: {self.tests_run}")
        print(f"Passed: {self.tests_passed}")
        print(f"Failed: {self.tests_run - self.tests_passed}")
        print(f"Success Rate: {(self.tests_passed/self.tests_run)*100:.1f}%")
        
        if self.failed_tests:
            print(f"\n❌ FAILED TESTS:")
            for i, test in enumerate(self.failed_tests, 1):
                print(f"{i}. {test['name']}")
                if 'expected' in test:
                    print(f"   Expected: {test['expected']}, Got: {test['actual']}")
                if 'error' in test:
                    print(f"   Error: {test['error']}")
                if 'response' in test:
                    print(f"   Response: {test['response']}")
                print(f"   URL: {test['url']}")
                print()

def main():
    print("🚀 Starting Quantum Flow Trading System API Tests")
    print("="*60)
    
    tester = QuantumFlowAPITester()
    
    # Run all tests
    print("\n🔧 Testing Core API Endpoints...")
    tester.test_root_endpoint()
    
    print("\n📊 Testing Data Endpoints...")
    tester.test_signals_endpoint()
    tester.test_active_signals_endpoint()
    tester.test_performance_endpoint()
    tester.test_config_endpoint()
    
    print("\n💹 Testing Market Data...")
    tester.test_market_data_endpoint()
    
    print("\n⚙️ Testing Configuration Updates...")
    tester.test_config_update()
    
    print("\n🚫 Testing Error Handling...")
    tester.test_invalid_endpoint()
    
    # Print final summary
    tester.print_summary()
    
    # Return appropriate exit code
    return 0 if tester.tests_passed == tester.tests_run else 1

if __name__ == "__main__":
    sys.exit(main())