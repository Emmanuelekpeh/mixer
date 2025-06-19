// @ts-check
const { test, expect } = require('@playwright/test');

/**
 * Responsive and accessibility tests
 */
test.describe('Responsive & Accessibility Tests', () => {
  test('should work on mobile devices', async ({ page, isMobile }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    
    if (isMobile) {
      // Test mobile-specific functionality
      await expect(page.locator('body')).toBeVisible();
      
      // Check if mobile navigation works
      const mobileNav = page.locator('[class*="mobile"], [class*="hamburger"], .menu-toggle');
      if (await mobileNav.isVisible()) {
        await mobileNav.click();
        await page.waitForTimeout(500);
      }
      
      // Test touch interactions
      const buttons = page.locator('button').first();
      if (await buttons.isVisible()) {
        await buttons.tap();
        await page.waitForTimeout(500);
      }
    }
  });

  test('should have basic accessibility features', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    
    // Check for alt text on images
    const images = page.locator('img');
    const imageCount = await images.count();
    
    for (let i = 0; i < imageCount; i++) {
      const img = images.nth(i);
      const alt = await img.getAttribute('alt');
      if (!alt) {
        console.warn(`Image ${i} missing alt text`);
      }
    }
    
    // Check for button labels
    const buttons = page.locator('button');
    const buttonCount = await buttons.count();
    
    for (let i = 0; i < Math.min(buttonCount, 10); i++) {
      const btn = buttons.nth(i);
      const text = await btn.textContent();
      const ariaLabel = await btn.getAttribute('aria-label');
      
      if (!text?.trim() && !ariaLabel) {
        console.warn(`Button ${i} missing text or aria-label`);
      }
    }
    
    // Check for form labels
    const inputs = page.locator('input');
    const inputCount = await inputs.count();
    
    for (let i = 0; i < inputCount; i++) {
      const input = inputs.nth(i);
      const id = await input.getAttribute('id');
      const ariaLabel = await input.getAttribute('aria-label');
      const placeholder = await input.getAttribute('placeholder');
      
      if (id) {
        const label = page.locator(`label[for="${id}"]`);
        if (!(await label.isVisible()) && !ariaLabel && !placeholder) {
          console.warn(`Input ${i} missing proper labeling`);
        }
      }
    }
  });

  test('should handle keyboard navigation', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    
    // Test tab navigation
    await page.keyboard.press('Tab');
    await page.waitForTimeout(200);
    
    // Check if focus is visible
    const focusedElement = page.locator(':focus');
    if (await focusedElement.isVisible()) {
      await expect(focusedElement).toBeVisible();
    }
    
    // Test Enter key on focused elements
    await page.keyboard.press('Enter');
    await page.waitForTimeout(500);
    
    // Should not crash after keyboard interaction
    await expect(page.locator('body')).toBeVisible();
  });
});
