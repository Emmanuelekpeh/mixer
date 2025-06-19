#!/usr/bin/env python3
"""
Performance Test - Check API response times
"""

import requests
import time
import threading

BASE_URL = "http://localhost:10000"

def test_single_request():
    """Test a single request timing"""
    start = time.time()
    response = requests.get(f"{BASE_URL}/api/models", timeout=5)
    end = time.time()
    
    return {
        "duration": end - start,
        "status": response.status_code,
        "success": response.status_code == 200
    }

def test_performance():
    """Test API performance"""
    print("⚡ PERFORMANCE TEST")
    print("=" * 40)
    
    # Test single request timing
    print("📊 Single Request Test...")
    result = test_single_request()
    print(f"   Duration: {result['duration']:.3f}s")
    print(f"   Status: {result['status']}")
    
    # Test multiple sequential requests
    print("\n📊 Sequential Requests Test...")
    start_time = time.time()
    successful = 0
    total = 10
    
    for i in range(total):
        try:
            response = requests.get(f"{BASE_URL}/api/models", timeout=3)
            if response.status_code == 200:
                successful += 1
        except:
            pass
    
    end_time = time.time()
    duration = end_time - start_time
    req_per_sec = successful / duration
    
    print(f"   Successful: {successful}/{total}")
    print(f"   Duration: {duration:.3f}s")
    print(f"   Speed: {req_per_sec:.1f} req/sec")
    
    # Performance assessment
    if req_per_sec > 10:
        print("   ✅ EXCELLENT performance!")
    elif req_per_sec > 5:
        print("   ✅ Good performance")
    elif req_per_sec > 2:
        print("   ⚠️ Moderate performance")
    else:
        print("   ❌ Slow performance - needs optimization")
    
    return req_per_sec

def test_concurrent_requests():
    """Test concurrent request handling"""
    print("\n🔄 Concurrent Requests Test...")
    
    results = []
    threads = []
    
    def make_request():
        try:
            start = time.time()
            response = requests.get(f"{BASE_URL}/api/models", timeout=5)
            end = time.time()
            results.append({
                "duration": end - start,
                "success": response.status_code == 200
            })
        except:
            results.append({"duration": 5, "success": False})
    
    # Launch 5 concurrent requests
    start_time = time.time()
    for i in range(5):
        thread = threading.Thread(target=make_request)
        threads.append(thread)
        thread.start()
    
    # Wait for all to complete
    for thread in threads:
        thread.join()
    
    end_time = time.time()
    
    successful = sum(1 for r in results if r["success"])
    avg_duration = sum(r["duration"] for r in results) / len(results)
    total_duration = end_time - start_time
    
    print(f"   Successful: {successful}/5")
    print(f"   Average response time: {avg_duration:.3f}s")
    print(f"   Total time: {total_duration:.3f}s")
    print(f"   Concurrent throughput: {successful/total_duration:.1f} req/sec")
    
    return successful == 5

if __name__ == "__main__":
    print("⚡ API PERFORMANCE ANALYSIS")
    print("=" * 50)
    
    sequential_speed = test_performance()
    concurrent_ok = test_concurrent_requests()
    
    print("\n" + "=" * 50)
    print("📋 PERFORMANCE SUMMARY")
    print("=" * 50)
    print(f"Sequential Speed: {sequential_speed:.1f} req/sec")
    print(f"Concurrent Handling: {'✅ Working' if concurrent_ok else '❌ Issues'}")
    
    if sequential_speed >= 5 and concurrent_ok:
        print("\n🚀 PERFORMANCE: EXCELLENT")
    elif sequential_speed >= 2:
        print("\n✅ PERFORMANCE: ACCEPTABLE")
    else:
        print("\n⚠️ PERFORMANCE: NEEDS IMPROVEMENT")
        print("   Suggestions:")
        print("   - Add database connection pooling")
        print("   - Cache frequently accessed data")
        print("   - Optimize database queries")
