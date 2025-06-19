// @ts-check
const { test, expect } = require('@playwright/test');

/**
 * Issue Detection Tests - specifically targeting known problems
 */
test.describe('Issue Detection Tests', () => {
  test('should identify model loading and path issues', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    
    // Monitor console for model-related errors
    const consoleLogs = [];
    const consoleErrors = [];
    
    page.on('console', msg => {
      const text = msg.text();
      consoleLogs.push(text);
      
      if (msg.type() === 'error') {
        consoleErrors.push(text);
      }
      
      // Look for specific model loading messages
      if (text.includes('model') || text.includes('Model') || text.includes('CNN') || text.includes('AST')) {
        console.log('🔍 Model-related log:', text);
      }
      
      // Look for path-related issues
      if (text.includes('path') || text.includes('directory') || text.includes('../models')) {
        console.log('📁 Path-related log:', text);
      }
    });
    
    // Try to start a mixing session to trigger model loading
    const nameInput = page.locator('input[placeholder*="Enter your name"]');
    if (await nameInput.isVisible()) {
      await nameInput.fill('TestUser');
      
      const fileInput = page.locator('input[type="file"]');
      if (await fileInput.isVisible()) {
        const testFile = Buffer.from('fake audio data');
        await fileInput.setInputFiles([{
          name: 'test.mp3',
          mimeType: 'audio/mpeg',
          buffer: testFile
        }]);
        
        await page.waitForTimeout(2000);
        
        const startBtn = page.locator('button:has-text("START MIXING SESSION")');
        if (await startBtn.isVisible()) {
          await startBtn.click();
          await page.waitForTimeout(5000); // Wait for model loading
        }
      }
    }
    
    console.log('🔍 Console errors found:', consoleErrors.length);
    console.log('📊 Total console logs:', consoleLogs.length);
    
    // Report findings
    const modelIssues = consoleLogs.filter(log => 
      log.includes('Models directory not found') || 
      log.includes('../models') ||
      log.includes('model') && log.toLowerCase().includes('error')
    );
    
    if (modelIssues.length > 0) {
      console.log('❌ Model loading issues detected:');
      modelIssues.forEach(issue => console.log('  -', issue));
    }
  });

  test('should check for tournament deletion capability', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    
    // Navigate through the app looking for tournament management
    const leaderboardLink = page.locator('text=LEADERBOARD');
    if (await leaderboardLink.isVisible()) {
      await leaderboardLink.click();
      await page.waitForLoadState('networkidle');
    }
    
    // Look for any saved tournaments or tournament lists
    const possibleTournamentElements = [
      '[class*="tournament"]',
      '[data-testid*="tournament"]', 
      '.tournament-item',
      '.tournament-card',
      '[id*="tournament"]',
      'li:has-text("tournament")',
      'div:has-text("tournament")'
    ];
    
    let foundTournaments = false;
    let foundDeleteButtons = false;
    
    for (const selector of possibleTournamentElements) {
      const elements = page.locator(selector);
      const count = await elements.count();
      if (count > 0) {
        console.log(`Found ${count} elements matching ${selector}`);
        foundTournaments = true;
        
        // Look for delete buttons near these elements
        const deleteSelectors = [
          'button:has-text("Delete")',
          'button:has-text("Remove")', 
          'button:has-text("×")',
          '[aria-label*="delete"]',
          '[title*="delete"]',
          '.delete-button',
          '.remove-button'
        ];
        
        for (const delSelector of deleteSelectors) {
          const delButtons = page.locator(delSelector);
          const delCount = await delButtons.count();
          if (delCount > 0) {
            foundDeleteButtons = true;
            console.log(`✅ Found ${delCount} delete buttons with selector: ${delSelector}`);
          }
        }
      }
    }
    
    if (foundTournaments && !foundDeleteButtons) {
      console.log('❌ ISSUE CONFIRMED: Found tournaments but NO delete functionality!');
      console.log('   This matches the user\'s report about not being able to delete sustained tournaments.');
    } else if (!foundTournaments) {
      console.log('ℹ️ No tournaments found in current UI state');
    } else {
      console.log('✅ Delete functionality appears to be available');
    }
  });

  test('should check API routing and model availability', async ({ page }) => {
    // Monitor all API calls
    const apiCalls = [];
    const failedCalls = [];
    
    page.on('response', response => {
      if (response.url().includes('localhost:10000') || response.url().includes('/api/')) {
        const callInfo = {
          url: response.url(),
          status: response.status(),
          method: response.request().method(),
          ok: response.ok()
        };
        
        apiCalls.push(callInfo);
        
        if (!response.ok()) {
          failedCalls.push(callInfo);
          console.log('❌ Failed API call:', callInfo);
        } else {
          console.log('✅ Successful API call:', callInfo);
        }
      }
    });
    
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    
    // Try to trigger various API calls
    const nameInput = page.locator('input[placeholder*="Enter your name"]');
    if (await nameInput.isVisible()) {
      await nameInput.fill('APITestUser');
      await page.waitForTimeout(1000);
    }
    
    await page.waitForTimeout(3000); // Wait for any background API calls
    
    console.log(`📊 Total API calls made: ${apiCalls.length}`);
    console.log(`❌ Failed API calls: ${failedCalls.length}`);
    
    // Check for specific endpoints that should exist
    const expectedEndpoints = [
      '/health',
      '/models',
      '/tournaments',
      '/api/tournaments',
      '/api/models'
    ];
    
    for (const endpoint of expectedEndpoints) {
      const found = apiCalls.some(call => call.url.includes(endpoint));
      if (found) {
        console.log(`✅ Endpoint ${endpoint} was called`);
      } else {
        console.log(`❓ Endpoint ${endpoint} was not accessed`);
      }
    }
    
    // Test a specific API endpoint directly
    try {
      const healthResponse = await page.request.get('http://localhost:10000/health');
      console.log(`🏥 Health endpoint status: ${healthResponse.status()}`);
      
      const modelsResponse = await page.request.get('http://localhost:10000/models');
      console.log(`🤖 Models endpoint status: ${modelsResponse.status()}`);
      
      if (modelsResponse.ok()) {
        const modelsData = await modelsResponse.json();
        console.log(`🤖 Available models: ${JSON.stringify(modelsData)}`);
      }
    } catch (error) {
      console.log('❌ Direct API test failed:', error.message);
    }
  });
});
