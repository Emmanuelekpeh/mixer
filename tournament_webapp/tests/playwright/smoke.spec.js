// @ts-check
const { test, expect } = require('@playwright/test');

/**
 * Basic smoke tests to ensure the application loads correctly
 */
test.describe('Tournament App - Smoke Tests', () => {
  test('should load the homepage', async ({ page }) => {
    await page.goto('/');
    
    // Check if the page loads without errors
    await expect(page).toHaveTitle(/Tournament|Mixer/i);
    
    // Check if key elements are present
    await expect(page.locator('body')).toBeVisible();
    
    // Check for React app initialization
    await page.waitForSelector('#root', { timeout: 10000 });
    await expect(page.locator('#root')).toBeVisible();
  });
  test('should show mixer setup form on first visit', async ({ page }) => {
    await page.goto('/');
    
    // Wait for React to load
    await page.waitForLoadState('networkidle');
    
    // Should show mixer setup form elements
    await expect(page.locator('text=ENTER YOUR ARTIST NAME')).toBeVisible({ timeout: 10000 });
    await expect(page.locator('text=UPLOAD YOUR AUDIO')).toBeVisible({ timeout: 10000 });
    await expect(page.locator('input[placeholder*="Enter your name"]')).toBeVisible();
    await expect(page.locator('input[type="file"]')).toBeVisible();
    await expect(page.locator('button:has-text("START MIXING SESSION")')).toBeVisible();
  });
  test('should handle navigation between routes', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
      // Test navigation to Leaderboard
    const leaderboardLink = page.locator('text=LEADERBOARD').or(page.locator('a:has-text("LEADERBOARD")'));
    
    if (await leaderboardLink.isVisible()) {
      await leaderboardLink.click();
      await page.waitForLoadState('networkidle');
        // Should not have error page
      await expect(page.locator('text=Error')).not.toBeVisible();
      await expect(page.locator('text=404')).not.toBeVisible();
      await expect(page.locator('text=Not Found')).not.toBeVisible();
      
      // Navigate back to home
      const homeLink = page.locator('text=HOME').or(page.locator('a:has-text("HOME")'));
      if (await homeLink.isVisible()) {
        await homeLink.click();
        await page.waitForLoadState('networkidle');
      }
    }
  });

  test('should check API connectivity', async ({ page }) => {
    await page.goto('/');
    
    // Listen for API calls
    const responses = [];
    page.on('response', response => {
      if (response.url().includes('localhost:10000') || response.url().includes('/api/')) {
        responses.push({
          url: response.url(),
          status: response.status(),
          ok: response.ok()
        });
      }
    });
    
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(2000); // Wait for any async API calls
    
    // Check if any API calls were made and if they succeeded
    console.log('API responses captured:', responses);
    
    // If API calls were made, ensure they didn't all fail
    if (responses.length > 0) {
      const successfulCalls = responses.filter(r => r.ok);
      expect(successfulCalls.length).toBeGreaterThan(0);
    }
  });
});
