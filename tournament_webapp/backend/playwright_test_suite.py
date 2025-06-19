#!/usr/bin/env python3
"""
🎭 Playwright UI Test Suite for Tournament Webapp
=================================================

Uses Playwright to test the actual user interface and identify data flow issues.
"""

import asyncio
import json
import time
from pathlib import Path

# Import the MCP Playwright tools
def test_tournament_ui_flow():
    """Test the complete tournament UI flow using Playwright"""
    print("🎭 Starting Playwright UI Tests...")
    
    try:
        # Use the MCP Playwright init browser tool
        from mcp_playwright_mc import init_browser, get_context, execute_code, get_screenshot
        
        # Initialize browser and navigate to tournament app
        print("🌐 Initializing browser...")
        browser_result = init_browser("http://localhost:3000")
        print(f"Browser result: {browser_result}")
        
        # Wait a moment for the page to load
        time.sleep(3)
        
        # Get page context
        print("📄 Getting page context...")
        context = get_context()
        print(f"Page context: {context}")
        
        # Take initial screenshot
        print("📸 Taking initial screenshot...")
        screenshot = get_screenshot()
        
        # Test tournament creation flow
        print("🏆 Testing tournament creation...")
        
        # Execute code to interact with the page
        ui_test_code = """
        async function run(page) {
            const logs = [];
            
            // Get current URL
            const url = page.url();
            logs.push(`Current URL: ${url}`);
            
            // Check if we're on the right page
            const title = await page.title();
            logs.push(`Page title: ${title}`);
            
            // Look for tournament setup elements
            const setupElements = await page.$$('[data-testid*="tournament"], [class*="Setup"], [class*="Tournament"]');
            logs.push(`Found ${setupElements.length} potential tournament setup elements`);
            
            // Check for file upload
            const fileInputs = await page.$$('input[type="file"]');
            logs.push(`Found ${fileInputs.length} file input elements`);
            
            // Check for any error messages or console errors
            const errorElements = await page.$$('[class*="error"], [class*="Error"]');
            logs.push(`Found ${errorElements.length} error elements`);
            
            // Check local storage for user/tournament data
            const localStorage = await page.evaluate(() => {
                return JSON.stringify(window.localStorage);
            });
            logs.push(`LocalStorage: ${localStorage}`);
            
            // Check for React DevTools or console errors
            const consoleErrors = [];
            page.on('console', msg => {
                if (msg.type() === 'error') {
                    consoleErrors.push(msg.text());
                }
            });
            
            return {
                url,
                title,
                setupElementCount: setupElements.length,
                fileInputCount: fileInputs.length,
                errorElementCount: errorElements.length,
                localStorage,
                logs
            };
        }
        """
        
        result = execute_code(ui_test_code)
        print(f"UI Test Result: {result}")
        
        # Test tournament data flow by checking for specific issues
        data_flow_test = """
        async function run(page) {
            const logs = [];
            
            // Check if we can access the React app state
            const reactData = await page.evaluate(() => {
                // Try to find React components and their state
                const reactFiber = document.querySelector('#root')._reactInternalInstance || 
                                 document.querySelector('#root').__reactInternalFiber;
                
                if (reactFiber) {
                    logs.push('React fiber found');
                    return 'React app detected';
                } else {
                    return 'No React fiber found';
                }
            });
            
            // Check for tournament data in console
            const tournamentLogs = await page.evaluate(() => {
                // Check if there are any tournament-related console logs
                return window.tournamentDebugData || 'No tournament debug data';
            });
            
            // Look for specific data structure issues mentioned in the logs
            const battleArenaElements = await page.$$('[class*="BattleArena"], [class*="Battle"]');
            logs.push(`Found ${battleArenaElements.length} battle arena elements`);
            
            return {
                reactData,
                tournamentLogs,
                battleArenaCount: battleArenaElements.length,
                logs
            };
        }
        """
        
        data_result = execute_code(data_flow_test)
        print(f"Data Flow Test Result: {data_result}")
        
        # Take final screenshot
        final_screenshot = get_screenshot()
        
        return {
            "status": "completed",
            "ui_test": result,
            "data_flow_test": data_result,
            "screenshots_taken": 2
        }
        
    except Exception as e:
        print(f"❌ Playwright test failed: {e}")
        return {"status": "failed", "error": str(e)}

# Also create a backend API test to run in parallel
def test_backend_api_detailed():
    """Detailed backend API testing"""
    print("🔧 Testing Backend API in detail...")
    
    import requests
    
    tests = {}
    
    try:
        # Test 1: Check if backend is running
        response = requests.get("http://localhost:10000/health", timeout=5)
        tests["backend_health"] = {
            "status": "pass" if response.status_code == 200 else "fail",
            "code": response.status_code
        }
        
        # Test 2: Get recent tournament to analyze data structure
        response = requests.get("http://localhost:10000/api/tournaments", timeout=5)
        if response.status_code == 200:
            data = response.json()
            tests["tournament_list"] = {
                "status": "pass",
                "tournament_count": len(data.get("tournaments", [])),
                "structure": list(data.keys()) if isinstance(data, dict) else "not_dict"
            }
            
            # Test 3: Get specific tournament details
            if data.get("tournaments") and len(data["tournaments"]) > 0:
                tournament_id = data["tournaments"][0].get("tournament_id") or data["tournaments"][0].get("id")
                if tournament_id:
                    response = requests.get(f"http://localhost:10000/api/tournaments/{tournament_id}", timeout=5)
                    if response.status_code == 200:
                        tournament_data = response.json()
                        tests["tournament_details"] = {
                            "status": "pass",
                            "has_pairs": "pairs" in tournament_data.get("tournament", {}),
                            "pairs_count": len(tournament_data.get("tournament", {}).get("pairs", [])),
                            "current_pair": tournament_data.get("tournament", {}).get("current_pair"),
                            "structure_keys": list(tournament_data.get("tournament", {}).keys())
                        }
        
        # Test 4: Check models endpoint
        response = requests.get("http://localhost:10000/api/models", timeout=5)
        if response.status_code == 200:
            models_data = response.json()
            tests["models_endpoint"] = {
                "status": "pass",
                "model_count": len(models_data.get("models", [])),
                "structure": list(models_data.keys()) if isinstance(models_data, dict) else "not_dict"
            }
        
    except Exception as e:
        tests["api_error"] = {"status": "fail", "error": str(e)}
    
    return tests

def main():
    """Main test runner"""
    print("🧪 Comprehensive Tournament Webapp Testing")
    print("=" * 50)
    
    # Wait for servers to be ready
    print("⏳ Waiting for servers to start...")
    time.sleep(5)
    
    # Test backend API
    api_results = test_backend_api_detailed()
    print("\n📊 Backend API Test Results:")
    for test, result in api_results.items():
        status_icon = "✅" if result.get("status") == "pass" else "❌"
        print(f"  {status_icon} {test}: {result}")
    
    # Test UI with Playwright
    print("\n🎭 Running Playwright UI Tests...")
    ui_results = test_tournament_ui_flow()
    print(f"\n📊 UI Test Results: {ui_results}")
    
    # Summary
    print("\n" + "=" * 50)
    print("🏁 COMPREHENSIVE TEST SUMMARY")
    print("=" * 50)
    
    api_passed = sum(1 for r in api_results.values() if r.get("status") == "pass")
    api_total = len(api_results)
    ui_status = ui_results.get("status", "unknown")
    
    print(f"📡 Backend API: {api_passed}/{api_total} tests passed")
    print(f"🖥️ Frontend UI: {ui_status}")
    
    if api_passed == api_total and ui_status == "completed":
        print("✅ ALL TESTS PASSED - System is healthy")
    else:
        print("⚠️ ISSUES DETECTED - Requires investigation")
    
    return {
        "api_results": api_results,
        "ui_results": ui_results,
        "summary": {
            "api_score": f"{api_passed}/{api_total}",
            "ui_status": ui_status
        }
    }

if __name__ == "__main__":
    results = main()
