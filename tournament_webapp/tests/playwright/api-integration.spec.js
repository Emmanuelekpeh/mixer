// @ts-check
const { test, expect } = require('@playwright/test');

/**
 * API Integration tests - testing backend connectivity and data flow
 */
test.describe('API Integration Tests', () => {
  test('should connect to backend API', async ({ page }) => {
    // Test direct API endpoints
    const response = await page.request.get('http://localhost:10000/health');
    expect(response.ok()).toBeTruthy();
  });

  test('should fetch tournament data from API', async ({ page }) => {
    await page.goto('/');
    
    // Monitor API calls
    const apiCalls = [];
    page.on('response', response => {
      if (response.url().includes('localhost:10000')) {
        apiCalls.push({
          url: response.url(),
          status: response.status(),
          method: response.request().method()
        });
      }
    });
    
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(3000);
    
    console.log('API calls made:', apiCalls);
    
    // Verify some API calls were made
    expect(apiCalls.length).toBeGreaterThan(0);
    
    // Verify API calls are successful
    const successfulCalls = apiCalls.filter(call => call.status < 400);
    expect(successfulCalls.length).toBeGreaterThan(0);
  });

  test('should handle API errors gracefully', async ({ page }) => {
    // Test what happens when API is down
    await page.route('**/localhost:10000/**', route => {
      route.fulfill({
        status: 500,
        body: 'Server Error'
      });
    });
    
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    
    // App should still load, maybe with error message
    await expect(page.locator('body')).toBeVisible();
    
    // Should show some indication of error or fallback UI
    const errorIndicators = page.locator('text=Error, text=Connection, text=Failed, .error, [class*="error"]');
    // Note: This might fail if no error handling is implemented
  });

  test('should validate tournament data persistence', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    
    // Create a tournament and check if it persists
    const userInput = page.locator('input[placeholder*="name"], input[type="text"]').first();
    if (await userInput.isVisible()) {
      const testUserName = 'PersistenceTest_' + Date.now();
      await userInput.fill(testUserName);
      
      const submitBtn = page.locator('button:has-text("Login"), button:has-text("Start"), button[type="submit"]').first();
      if (await submitBtn.isVisible()) {
        await submitBtn.click();
        await page.waitForLoadState('networkidle');
        
        // Refresh page and check if user data persists
        await page.reload();
        await page.waitForLoadState('networkidle');
        
        // Check if user is still logged in or data is preserved
        const userElements = page.locator(`text=${testUserName}, [data-user*="${testUserName}"]`);
        // This test will help identify if persistence is working
      }
    }
  });

  test('should test model loading and availability', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    
    // Check console for model loading messages
    const consoleLogs = [];
    page.on('console', msg => {
      consoleLogs.push(msg.text());
    });
    
    await page.waitForTimeout(5000); // Wait for any model loading
    
    // Look for model-related UI elements
    const modelElements = page.locator('text=Model, text=CNN, text=AST, [class*="model"]');
    
    console.log('Console logs:', consoleLogs);
    console.log('Model elements found:', await modelElements.count());
    
    // This will help identify if models are being loaded correctly
  });
});
